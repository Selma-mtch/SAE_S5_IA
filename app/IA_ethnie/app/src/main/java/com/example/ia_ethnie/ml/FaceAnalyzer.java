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
    private Interpreter transferInterpreter;

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

            try {
                transferInterpreter = new Interpreter(
                        loadModelFile(context, "model_transfer.tflite"), options);
            } catch (IOException e) {
                // Modèle transfer non disponible
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

        switch (currentModelType) {
            case SPECIALIZED:
                return analyzeWithSpecializedModels(resizedBitmap);
            case TRANSFER:
                return analyzeWithTransferModel(preprocessRGB(resizedBitmap));
            case MULTI_TASK:
            default:
                return analyzeWithMultiTaskModel(preprocessGrayscale(resizedBitmap));
        }
    }

    /**
     * Preprocessing grayscale [0, 1] pour MULTI_TASK et SPECIALIZED.
     * 1 canal, shape: (1, 128, 128, 1)
     */
    private ByteBuffer preprocessGrayscale(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            buffer.putFloat(gray);
        }

        buffer.rewind();
        return buffer;
    }

    /**
     * Preprocessing RGB [-1, 1] pour TRANSFER (EfficientNet preprocess_input).
     * 3 canaux, shape: (1, 128, 128, 3)
     */
    private ByteBuffer preprocessRGB(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            // EfficientNet preprocess_input : normalise en [-1, 1]
            buffer.putFloat(r / 127.5f - 1.0f);
            buffer.putFloat(g / 127.5f - 1.0f);
            buffer.putFloat(b / 127.5f - 1.0f);
        }

        buffer.rewind();
        return buffer;
    }

    private PredictionResult analyzeWithMultiTaskModel(ByteBuffer inputBuffer) {
        if (multiTaskInterpreter == null) {
            return createDemoResult();
        }

        // Detecter les sorties par leur NOM de tenseur (contient "age", "gender", "ethnicity")
        // et en fallback par leur shape
        int outputCount = multiTaskInterpreter.getOutputTensorCount();

        int ageIdx = -1, genderIdx = -1, ethIdx = -1;

        // D'abord essayer par nom
        for (int i = 0; i < outputCount; i++) {
            String name = multiTaskInterpreter.getOutputTensor(i).name().toLowerCase();
            if (name.contains("age")) {
                ageIdx = i;
            } else if (name.contains("gender")) {
                genderIdx = i;
            } else if (name.contains("ethnic")) {
                ethIdx = i;
            }
        }

        // Fallback par shape si les noms n'ont pas marche
        if (ageIdx == -1 || genderIdx == -1 || ethIdx == -1) {
            for (int i = 0; i < outputCount; i++) {
                int[] shape = multiTaskInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == NUM_ETHNICITY_CLASSES && ethIdx == -1) {
                    ethIdx = i;
                }
            }
            // Les deux (1,1) restants : on log les valeurs pour debug
            for (int i = 0; i < outputCount; i++) {
                if (i == ethIdx) continue;
                int[] shape = multiTaskInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == 1) {
                    if (ageIdx == -1) ageIdx = i;
                    else if (genderIdx == -1) genderIdx = i;
                }
            }
        }

        if (ageIdx == -1) ageIdx = 0;
        if (genderIdx == -1) genderIdx = 1;
        if (ethIdx == -1) ethIdx = 2;

        // Log pour debug
        android.util.Log.d("FaceAnalyzer", "Output mapping -> age=" + ageIdx
                + " gender=" + genderIdx + " ethnicity=" + ethIdx);
        for (int i = 0; i < outputCount; i++) {
            android.util.Log.d("FaceAnalyzer", "Output " + i
                    + ": name=" + multiTaskInterpreter.getOutputTensor(i).name()
                    + " shape=" + java.util.Arrays.toString(multiTaskInterpreter.getOutputTensor(i).shape()));
        }

        float[][] ageOutput = new float[1][1];
        float[][] genderOutput = new float[1][1];
        float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];

        Object[] inputs = {inputBuffer};
        java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
        outputs.put(ageIdx, ageOutput);
        outputs.put(genderIdx, genderOutput);
        outputs.put(ethIdx, ethnicityOutput);

        multiTaskInterpreter.runForMultipleInputsOutputs(inputs, outputs);

        android.util.Log.d("FaceAnalyzer", "Raw age=" + ageOutput[0][0]
                + " gender=" + genderOutput[0][0]
                + " ethnicity=" + java.util.Arrays.toString(ethnicityOutput[0]));

        // Si age ressemble a une sigmoid (0-1) et gender a un age (>1),
        // ils sont inverses -> on swap
        if (ageOutput[0][0] >= 0.0f && ageOutput[0][0] <= 1.0f
                && genderOutput[0][0] > 1.0f) {
            android.util.Log.w("FaceAnalyzer", "Age/Gender swapped, correcting...");
            float[][] temp = ageOutput;
            ageOutput = genderOutput;
            genderOutput = temp;
        }

        // Age : valeur brute de la regression
        int predictedAge = Math.round(ageOutput[0][0]);

        // Gender : sigmoid -> 0 = Homme, 1 = Femme
        float genderSigmoid = genderOutput[0][0];
        int genderIndex = genderSigmoid >= 0.5f ? 1 : 0;
        float genderConf = genderIndex == 1 ? genderSigmoid : (1.0f - genderSigmoid);

        // Ethnicity : softmax -> argmax
        int ethnicityIndex = argMax(ethnicityOutput[0]);

        return new PredictionResult(
                Math.max(0, Math.min(100, predictedAge)),
                GENDER_LABELS[genderIndex],
                ETHNICITY_LABELS[ethnicityIndex],
                1.0f,
                genderConf,
                ethnicityOutput[0][ethnicityIndex],
                ModelType.MULTI_TASK
        );
    }

    private PredictionResult analyzeWithTransferModel(ByteBuffer inputBuffer) {
        if (transferInterpreter == null) {
            return createDemoResult();
        }

        int outputCount = transferInterpreter.getOutputTensorCount();

        int ageIdx = -1, genderIdx = -1, ethIdx = -1;

        for (int i = 0; i < outputCount; i++) {
            String name = transferInterpreter.getOutputTensor(i).name().toLowerCase();
            if (name.contains("age")) ageIdx = i;
            else if (name.contains("gender")) genderIdx = i;
            else if (name.contains("ethnic")) ethIdx = i;
        }

        if (ageIdx == -1 || genderIdx == -1 || ethIdx == -1) {
            for (int i = 0; i < outputCount; i++) {
                int[] shape = transferInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == NUM_ETHNICITY_CLASSES && ethIdx == -1) {
                    ethIdx = i;
                }
            }
            for (int i = 0; i < outputCount; i++) {
                if (i == ethIdx) continue;
                int[] shape = transferInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == 1) {
                    if (ageIdx == -1) ageIdx = i;
                    else if (genderIdx == -1) genderIdx = i;
                }
            }
        }

        if (ageIdx == -1) ageIdx = 0;
        if (genderIdx == -1) genderIdx = 1;
        if (ethIdx == -1) ethIdx = 2;

        float[][] ageOutput = new float[1][1];
        float[][] genderOutput = new float[1][1];
        float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];

        Object[] inputs = {inputBuffer};
        java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
        outputs.put(ageIdx, ageOutput);
        outputs.put(genderIdx, genderOutput);
        outputs.put(ethIdx, ethnicityOutput);

        transferInterpreter.runForMultipleInputsOutputs(inputs, outputs);

        android.util.Log.d("FaceAnalyzer", "[TRANSFER] Raw age=" + ageOutput[0][0]
                + " gender=" + genderOutput[0][0]
                + " ethnicity=" + java.util.Arrays.toString(ethnicityOutput[0]));

        if (ageOutput[0][0] >= 0.0f && ageOutput[0][0] <= 1.0f
                && genderOutput[0][0] > 1.0f) {
            float[][] temp = ageOutput;
            ageOutput = genderOutput;
            genderOutput = temp;
        }

        int predictedAge = Math.round(ageOutput[0][0]);
        float genderSigmoid = genderOutput[0][0];
        int genderIndex = genderSigmoid >= 0.5f ? 1 : 0;
        float genderConf = genderIndex == 1 ? genderSigmoid : (1.0f - genderSigmoid);
        int ethnicityIndex = argMax(ethnicityOutput[0]);

        return new PredictionResult(
                Math.max(0, Math.min(100, predictedAge)),
                GENDER_LABELS[genderIndex],
                ETHNICITY_LABELS[ethnicityIndex],
                1.0f,
                genderConf,
                ethnicityOutput[0][ethnicityIndex],
                ModelType.TRANSFER
        );
    }

    private PredictionResult analyzeWithSpecializedModels(Bitmap bitmap) {
        int predictedAge = 25;
        float ageConfidence = 0.0f;
        int genderIndex = 0;
        float genderConfidence = 0.0f;
        int ethnicityIndex = 0;
        float ethnicityConfidence = 0.0f;

        // Modèle âge (RGB [-1, 1])
        if (ageInterpreter != null) {
            ByteBuffer rgbBuffer = preprocessRGB(bitmap);
            float[][] ageOutput = new float[1][1];
            ageInterpreter.run(rgbBuffer, ageOutput);
            predictedAge = Math.round(ageOutput[0][0]);
            ageConfidence = 1.0f;
        }

        // Modèle genre (grayscale [0, 1])
        if (genderInterpreter != null) {
            ByteBuffer grayBuffer = preprocessGrayscale(bitmap);
            float[][] genderOutput = new float[1][NUM_GENDER_CLASSES];
            genderInterpreter.run(grayBuffer, genderOutput);
            genderIndex = argMax(genderOutput[0]);
            genderConfidence = genderOutput[0][genderIndex];
        }

        // Modèle ethnicité (grayscale [0, 1])
        if (ethnicityInterpreter != null) {
            ByteBuffer grayBuffer = preprocessGrayscale(bitmap);
            float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];
            ethnicityInterpreter.run(grayBuffer, ethnicityOutput);
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
        if (transferInterpreter != null) transferInterpreter.close();
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
