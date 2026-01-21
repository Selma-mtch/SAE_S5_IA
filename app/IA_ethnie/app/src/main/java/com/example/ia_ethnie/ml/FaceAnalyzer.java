package com.example.ia_ethnie.ml;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FaceAnalyzer {
    private static final int INPUT_SIZE = 128;
    private static final int NUM_ETHNICITY_CLASSES = 5;
    private static final int NUM_GENDER_CLASSES = 2;

    private Interpreter ethnicityInterpreter;
    private Interpreter ageInterpreter;
    private Interpreter genderInterpreter;
    private Interpreter multiTaskInterpreter;

    private GpuDelegate gpuDelegate;
    private boolean useGpu = false;
    private ModelType currentModelType = ModelType.MULTI_TASK;

    public static final String[] ETHNICITY_LABELS = {"Blanc", "Noir", "Asiatique", "Indien", "Autre"};
    public static final String[] GENDER_LABELS = {"Homme", "Femme"};

    public enum ModelType {
        SPECIALIZED,    // 3 modèles séparés
        MULTI_TASK,     // 1 modèle multi-tâches
        TRANSFER        // MobileNetV2/EfficientNet
    }

    public static class PredictionResult {
        public int age;
        public String gender;
        public String ethnicity;
        public float ageConfidence;
        public float genderConfidence;
        public float ethnicityConfidence;
        public ModelType modelType;

        public PredictionResult(int age, String gender, String ethnicity,
                                float ageConfidence, float genderConfidence,
                                float ethnicityConfidence, ModelType modelType) {
            this.age = age;
            this.gender = gender;
            this.ethnicity = ethnicity;
            this.ageConfidence = ageConfidence;
            this.genderConfidence = genderConfidence;
            this.ethnicityConfidence = ethnicityConfidence;
            this.modelType = modelType;
        }
    }

    public FaceAnalyzer(Context context) {
        loadModels(context);
    }

    private void loadModels(Context context) {
        try {
            Interpreter.Options options = new Interpreter.Options();

            if (useGpu) {
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            }
            options.setNumThreads(4);

            // Charger les modèles depuis assets
            // Les fichiers .tflite doivent être placés dans app/src/main/assets/
            try {
                multiTaskInterpreter = new Interpreter(
                        loadModelFile(context, "model_multitask.tflite"), options);
            } catch (IOException e) {
                // Modèle multi-tâche non disponible
            }

            try {
                ethnicityInterpreter = new Interpreter(
                        loadModelFile(context, "model_ethnicity.tflite"), options);
            } catch (IOException e) {
                // Modèle ethnicité non disponible
            }

            try {
                ageInterpreter = new Interpreter(
                        loadModelFile(context, "model_age.tflite"), options);
            } catch (IOException e) {
                // Modèle âge non disponible
            }

            try {
                genderInterpreter = new Interpreter(
                        loadModelFile(context, "model_gender.tflite"), options);
            } catch (IOException e) {
                // Modèle genre non disponible
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        FileInputStream fis = new FileInputStream(
                context.getAssets().openFd(modelName).getFileDescriptor());
        FileChannel fileChannel = fis.getChannel();
        long startOffset = context.getAssets().openFd(modelName).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelName).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void setModelType(ModelType type) {
        this.currentModelType = type;
    }

    public ModelType getModelType() {
        return currentModelType;
    }

    public PredictionResult analyze(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer inputBuffer = preprocessImage(resizedBitmap);

        switch (currentModelType) {
            case SPECIALIZED:
                return analyzeWithSpecializedModels(inputBuffer);
            case TRANSFER:
            case MULTI_TASK:
            default:
                return analyzeWithMultiTaskModel(inputBuffer);
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            // Conversion en niveaux de gris et normalisation [0, 1]
            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            buffer.putFloat(gray);
        }

        buffer.rewind();
        return buffer;
    }

    private PredictionResult analyzeWithMultiTaskModel(ByteBuffer inputBuffer) {
        if (multiTaskInterpreter == null) {
            // Mode démo si pas de modèle
            return createDemoResult();
        }

        // Sorties pour le modèle multi-tâches
        float[][] ageOutput = new float[1][1];
        float[][] genderOutput = new float[1][NUM_GENDER_CLASSES];
        float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];

        Object[] inputs = {inputBuffer};
        java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
        outputs.put(0, ageOutput);
        outputs.put(1, genderOutput);
        outputs.put(2, ethnicityOutput);

        multiTaskInterpreter.runForMultipleInputsOutputs(inputs, outputs);

        int predictedAge = Math.round(ageOutput[0][0]);
        int genderIndex = argMax(genderOutput[0]);
        int ethnicityIndex = argMax(ethnicityOutput[0]);

        return new PredictionResult(
                Math.max(0, Math.min(100, predictedAge)),
                GENDER_LABELS[genderIndex],
                ETHNICITY_LABELS[ethnicityIndex],
                1.0f, // L'âge est une régression
                genderOutput[0][genderIndex],
                ethnicityOutput[0][ethnicityIndex],
                ModelType.MULTI_TASK
        );
    }

    private PredictionResult analyzeWithSpecializedModels(ByteBuffer inputBuffer) {
        int predictedAge = 25;
        float ageConfidence = 0.0f;
        int genderIndex = 0;
        float genderConfidence = 0.0f;
        int ethnicityIndex = 0;
        float ethnicityConfidence = 0.0f;

        // Modèle âge
        if (ageInterpreter != null) {
            float[][] ageOutput = new float[1][1];
            inputBuffer.rewind();
            ageInterpreter.run(inputBuffer, ageOutput);
            predictedAge = Math.round(ageOutput[0][0]);
            ageConfidence = 1.0f;
        }

        // Modèle genre
        if (genderInterpreter != null) {
            float[][] genderOutput = new float[1][NUM_GENDER_CLASSES];
            inputBuffer.rewind();
            genderInterpreter.run(inputBuffer, genderOutput);
            genderIndex = argMax(genderOutput[0]);
            genderConfidence = genderOutput[0][genderIndex];
        }

        // Modèle ethnicité
        if (ethnicityInterpreter != null) {
            float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];
            inputBuffer.rewind();
            ethnicityInterpreter.run(inputBuffer, ethnicityOutput);
            ethnicityIndex = argMax(ethnicityOutput[0]);
            ethnicityConfidence = ethnicityOutput[0][ethnicityIndex];
        }

        if (ageInterpreter == null && genderInterpreter == null && ethnicityInterpreter == null) {
            return createDemoResult();
        }

        return new PredictionResult(
                Math.max(0, Math.min(100, predictedAge)),
                GENDER_LABELS[genderIndex],
                ETHNICITY_LABELS[ethnicityIndex],
                ageConfidence,
                genderConfidence,
                ethnicityConfidence,
                ModelType.SPECIALIZED
        );
    }

    private PredictionResult createDemoResult() {
        // Résultat de démonstration quand aucun modèle n'est chargé
        return new PredictionResult(
                25,
                "Homme",
                "Blanc",
                0.85f,
                0.92f,
                0.78f,
                currentModelType
        );
    }

    private int argMax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void close() {
        if (ethnicityInterpreter != null) ethnicityInterpreter.close();
        if (ageInterpreter != null) ageInterpreter.close();
        if (genderInterpreter != null) genderInterpreter.close();
        if (multiTaskInterpreter != null) multiTaskInterpreter.close();
        if (gpuDelegate != null) gpuDelegate.close();
    }

    public String getModelInfo() {
        StringBuilder info = new StringBuilder();
        info.append("Type de modèle: ").append(currentModelType.name()).append("\n\n");

        switch (currentModelType) {
            case SPECIALIZED:
                info.append("3 modèles spécialisés:\n");
                info.append("- Modèle âge: CNN avec régression\n");
                info.append("- Modèle genre: CNN binaire\n");
                info.append("- Modèle ethnicité: CNN 5 classes\n");
                break;
            case MULTI_TASK:
                info.append("1 modèle multi-tâches:\n");
                info.append("- Architecture: ResNet + SE-Net\n");
                info.append("- Entrée: 128x128 niveaux de gris\n");
                info.append("- Sorties: âge, genre, ethnicité\n");
                break;
            case TRANSFER:
                info.append("Modèle par transfert:\n");
                info.append("- Base: MobileNetV2/EfficientNetB0\n");
                info.append("- Fine-tuning sur UTKFace\n");
                break;
        }

        info.append("\nClasses ethnicité:\n");
        for (String label : ETHNICITY_LABELS) {
            info.append("- ").append(label).append("\n");
        }

        return info.toString();
    }
}
