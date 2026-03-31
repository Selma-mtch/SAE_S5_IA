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
    private FaceDetectorHelper faceDetectorHelper;
    private volatile boolean closed = false;

    // Buffers réutilisables pour éviter les allocations à chaque frame
    private ByteBuffer grayscaleBuffer;
    private ByteBuffer rgbBuffer;

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
        public Bitmap croppedFace;
        public android.graphics.Rect faceBounds;

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
        this(context, false);
    }

    public FaceAnalyzer(Context context, boolean fastFaceDetection) {
        grayscaleBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4);
        grayscaleBuffer.order(ByteOrder.nativeOrder());
        rgbBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        rgbBuffer.order(ByteOrder.nativeOrder());
        faceDetectorHelper = new FaceDetectorHelper(fastFaceDetection);
        loadModels(context);
    }

    private void loadModels(Context context) {
        Interpreter.Options options = new Interpreter.Options();

        try {
            if (useGpu) {
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            }
        } catch (Exception e) {
            android.util.Log.w("FaceAnalyzer", "GPU delegate non disponible, fallback CPU", e);
        }
        options.setNumThreads(4);

        // Charger les modèles depuis assets (chaque modèle indépendant)
        try {
            multiTaskInterpreter = new Interpreter(
                    loadModelFile(context, "model_multitask.tflite"), options);
            android.util.Log.i("FaceAnalyzer", "model_multitask.tflite chargé");
        } catch (Exception e) {
            android.util.Log.w("FaceAnalyzer", "model_multitask.tflite non disponible: " + e.getMessage());
        }

        try {
            ethnicityInterpreter = new Interpreter(
                    loadModelFile(context, "model_ethnicity.tflite"), options);
            android.util.Log.i("FaceAnalyzer", "model_ethnicity.tflite chargé");
        } catch (Exception e) {
            android.util.Log.w("FaceAnalyzer", "model_ethnicity.tflite non disponible: " + e.getMessage());
        }

        try {
            ageInterpreter = new Interpreter(
                    loadModelFile(context, "model_age.tflite"), options);
            android.util.Log.i("FaceAnalyzer", "model_age.tflite chargé");
        } catch (Exception e) {
            android.util.Log.w("FaceAnalyzer", "model_age.tflite non disponible: " + e.getMessage());
        }

        try {
            genderInterpreter = new Interpreter(
                    loadModelFile(context, "model_gender.tflite"), options);
            android.util.Log.i("FaceAnalyzer", "model_gender.tflite chargé");
        } catch (Exception e) {
            android.util.Log.w("FaceAnalyzer", "model_gender.tflite non disponible: " + e.getMessage());
        }

        try {
            transferInterpreter = new Interpreter(
                    loadModelFile(context, "model_transfer.tflite"), options);
            android.util.Log.i("FaceAnalyzer", "model_transfer.tflite chargé");
        } catch (Exception e) {
            android.util.Log.e("FaceAnalyzer", "model_transfer.tflite ERREUR: " + e.getMessage(), e);
        }

    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        android.content.res.AssetFileDescriptor afd = context.getAssets().openFd(modelName);
        try (FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            FileChannel fileChannel = fis.getChannel();
            long startOffset = afd.getStartOffset();
            long declaredLength = afd.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } finally {
            afd.close();
        }
    }

    public void setModelType(ModelType type) {
        this.currentModelType = type;
    }

    public ModelType getModelType() {
        return currentModelType;
    }

    public synchronized PredictionResult analyze(Bitmap bitmap) {
        if (closed) return createDemoResult();

        // Detecter et cropper le visage avant l'inference
        FaceDetectorHelper.DetectionResult detection = faceDetectorHelper.detectAndCropWithBounds(bitmap);
        Bitmap faceBitmap = detection.croppedFace;
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true);

        PredictionResult result;
        switch (currentModelType) {
            case SPECIALIZED:
                result = analyzeWithSpecializedModels(resizedBitmap);
                break;
            case TRANSFER:
                result = analyzeWithTransferModel(preprocessRGB(resizedBitmap));
                break;
            case MULTI_TASK:
            default:
                result = analyzeWithMultiTaskModel(preprocessGrayscale(resizedBitmap));
                break;
        }

        // Stocker le crop et les bounds dans le resultat
        result.croppedFace = faceBitmap;
        result.faceBounds = detection.faceBounds;

        if (resizedBitmap != faceBitmap) {
            resizedBitmap.recycle();
        }
        return result;
    }

    /**
     * Preprocessing grayscale [0, 1] pour MULTI_TASK et SPECIALIZED.
     * 1 canal, shape: (1, 128, 128, 1)
     */
    private ByteBuffer preprocessGrayscale(Bitmap bitmap) {
        grayscaleBuffer.rewind();

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            grayscaleBuffer.putFloat(gray);
        }

        grayscaleBuffer.rewind();
        return grayscaleBuffer;
    }

    /**
     * Preprocessing RGB [-1, 1] pour TRANSFER (EfficientNet preprocess_input).
     * 3 canaux, shape: (1, 128, 128, 3)
     */
    private ByteBuffer preprocessRGB(Bitmap bitmap) {
        rgbBuffer.rewind();

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            // MobileNetV2 preprocess_input : normalise en [-1, 1]
            rgbBuffer.putFloat(r / 127.5f - 1.0f);
            rgbBuffer.putFloat(g / 127.5f - 1.0f);
            rgbBuffer.putFloat(b / 127.5f - 1.0f);
        }

        rgbBuffer.rewind();
        return rgbBuffer;
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
            android.util.Log.e("FaceAnalyzer", "Transfer interpreter is null!");
            return createDemoResult();
        }

        int outputCount = transferInterpreter.getOutputTensorCount();
        android.util.Log.d("FaceAnalyzer", "Transfer model output count: " + outputCount);

        int ageIdx = -1, genderIdx = -1, ethIdx = -1;

        // D'abord par nom de tenseur (ex: "age", "gender", "ethnic"/"race")
        for (int i = 0; i < outputCount; i++) {
            String name = transferInterpreter.getOutputTensor(i).name().toLowerCase();
            int[] shape = transferInterpreter.getOutputTensor(i).shape();
            android.util.Log.d("FaceAnalyzer", "Output " + i + ": name='" + name + "', shape=" + java.util.Arrays.toString(shape));
            if (name.contains("age")) ageIdx = i;
            else if (name.contains("gender")) genderIdx = i;
            else if (name.contains("ethnic") || name.contains("race")) ethIdx = i;
        }

        // Fallback par suffixe du nom TFLite (ex: "StatefulPartitionedCall:0" → :0=age, :1=gender, :2=eth)
        if (ageIdx == -1 || genderIdx == -1 || ethIdx == -1) {
            for (int i = 0; i < outputCount; i++) {
                String name = transferInterpreter.getOutputTensor(i).name();
                int colonIdx = name.lastIndexOf(':');
                if (colonIdx >= 0) {
                    try {
                        int suffix = Integer.parseInt(name.substring(colonIdx + 1));
                        if (suffix == 0 && ageIdx == -1) ageIdx = i;
                        else if (suffix == 1 && genderIdx == -1) genderIdx = i;
                        else if (suffix == 2 && ethIdx == -1) ethIdx = i;
                    } catch (NumberFormatException ignored) {}
                }
            }
        }

        // Fallback par shape: ethnie=[1,5], genre=[1,2], age=[1,1]
        if (ageIdx == -1 || genderIdx == -1 || ethIdx == -1) {
            for (int i = 0; i < outputCount; i++) {
                int[] shape = transferInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == NUM_ETHNICITY_CLASSES && ethIdx == -1) {
                    ethIdx = i;
                } else if (shape.length == 2 && shape[1] == NUM_GENDER_CLASSES && genderIdx == -1) {
                    genderIdx = i;
                }
            }
            for (int i = 0; i < outputCount; i++) {
                if (i == ethIdx || i == genderIdx) continue;
                int[] shape = transferInterpreter.getOutputTensor(i).shape();
                if (shape.length == 2 && shape[1] == 1 && ageIdx == -1) {
                    ageIdx = i;
                }
            }
        }

        if (ageIdx == -1) ageIdx = 0;
        if (genderIdx == -1) genderIdx = 1;
        if (ethIdx == -1) ethIdx = 2;

        // Déterminer la shape du gender output (softmax 2 classes ou sigmoid 1 classe)
        int genderOutputSize = transferInterpreter.getOutputTensor(genderIdx).shape()[1];

        float[][] ageOutput = new float[1][1];
        float[][] genderOutput = new float[1][genderOutputSize];
        float[][] ethnicityOutput = new float[1][NUM_ETHNICITY_CLASSES];

        Object[] inputs = {inputBuffer};
        java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
        outputs.put(ageIdx, ageOutput);
        outputs.put(genderIdx, genderOutput);
        outputs.put(ethIdx, ethnicityOutput);

        transferInterpreter.runForMultipleInputsOutputs(inputs, outputs);

        android.util.Log.d("FaceAnalyzer", "Transfer raw outputs - age=" + ageOutput[0][0]
                + " gender=" + java.util.Arrays.toString(genderOutput[0])
                + " eth=" + java.util.Arrays.toString(ethnicityOutput[0]));
        android.util.Log.d("FaceAnalyzer", "Index mapping - ageIdx=" + ageIdx + " genderIdx=" + genderIdx + " ethIdx=" + ethIdx);

        // Age
        int predictedAge = Math.round(ageOutput[0][0]);

        // Gender : softmax 2 classes ou sigmoid
        int genderIndex;
        float genderConf;
        if (genderOutputSize >= NUM_GENDER_CLASSES) {
            // Softmax 2 classes [Homme, Femme]
            genderIndex = argMax(genderOutput[0]);
            genderConf = genderOutput[0][genderIndex];
        } else {
            // Sigmoid : 0 = Homme, 1 = Femme
            float genderSigmoid = genderOutput[0][0];
            genderIndex = genderSigmoid >= 0.5f ? 1 : 0;
            genderConf = genderIndex == 1 ? genderSigmoid : (1.0f - genderSigmoid);
        }

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
            // Si la sortie est normalisée [0,1], dé-normaliser en multipliant par 100
            float ageRaw = ageOutput[0][0];
            predictedAge = Math.round(ageRaw <= 1.0f ? ageRaw * 100.0f : ageRaw);
            ageConfidence = 1.0f;
        }

        // Modèle genre (grayscale [0, 1])
        if (genderInterpreter != null) {
            ByteBuffer grayBuffer = preprocessGrayscale(bitmap);
            int genderOutputSize = genderInterpreter.getOutputTensor(0).shape()[1];
            float[][] genderOutput = new float[1][genderOutputSize];
            genderInterpreter.run(grayBuffer, genderOutput);
            if (genderOutputSize >= NUM_GENDER_CLASSES) {
                // Softmax 2 classes
                genderIndex = argMax(genderOutput[0]);
                genderConfidence = genderOutput[0][genderIndex];
            } else {
                // Sigmoid 1 classe : >0.5 = Femme
                float sig = genderOutput[0][0];
                genderIndex = sig >= 0.5f ? 1 : 0;
                genderConfidence = genderIndex == 1 ? sig : (1.0f - sig);
            }
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
        // Résultat par défaut quand aucun modèle n'est chargé
        return new PredictionResult(
                0,
                "N/A",
                "N/A",
                0.0f,
                0.0f,
                0.0f,
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

    public synchronized void close() {
        closed = true;
        if (ethnicityInterpreter != null) ethnicityInterpreter.close();
        if (ageInterpreter != null) ageInterpreter.close();
        if (genderInterpreter != null) genderInterpreter.close();
        if (multiTaskInterpreter != null) multiTaskInterpreter.close();
        if (transferInterpreter != null) transferInterpreter.close();
        if (gpuDelegate != null) gpuDelegate.close();
        if (faceDetectorHelper != null) faceDetectorHelper.close();
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
