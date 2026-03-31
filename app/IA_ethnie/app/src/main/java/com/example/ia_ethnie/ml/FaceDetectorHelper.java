package com.example.ia_ethnie.ml;

import android.graphics.Bitmap;
import android.graphics.Rect;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.util.List;

/**
 * Helper pour la detection de visage avec Google ML Kit.
 * Detecte le visage principal et retourne un Bitmap croppe.
 */
public class FaceDetectorHelper {

    private final FaceDetector detector;
    private static final float MARGIN_RATIO = 0.2f; // 20% de marge autour du visage

    /** Resultat de la detection : bitmap croppe + bounds sur l'image originale */
    public static class DetectionResult {
        public final Bitmap croppedFace;
        public final Rect faceBounds; // null si aucun visage detecte

        public DetectionResult(Bitmap croppedFace, Rect faceBounds) {
            this.croppedFace = croppedFace;
            this.faceBounds = faceBounds;
        }
    }

    public FaceDetectorHelper(boolean fastMode) {
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(fastMode
                        ? FaceDetectorOptions.PERFORMANCE_MODE_FAST
                        : FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setMinFaceSize(0.15f)
                .build();

        detector = FaceDetection.getClient(options);
    }

    /**
     * Detecte et croppe le visage principal (synchrone).
     * A appeler depuis un thread en arriere-plan, jamais le main thread.
     * Retourne l'image originale si aucun visage detecte.
     */
    public Bitmap detectAndCrop(Bitmap bitmap) {
        return detectAndCropWithBounds(bitmap).croppedFace;
    }

    /**
     * Detecte et croppe le visage principal, retourne aussi les bounds.
     */
    public DetectionResult detectAndCropWithBounds(Bitmap bitmap) {
        try {
            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);
            List<Face> faces = Tasks.await(detector.process(inputImage));

            if (faces == null || faces.isEmpty()) {
                return new DetectionResult(bitmap, null);
            }

            // Prendre le plus grand visage
            Face bestFace = faces.get(0);
            for (Face face : faces) {
                if (face.getBoundingBox().width() * face.getBoundingBox().height()
                        > bestFace.getBoundingBox().width() * bestFace.getBoundingBox().height()) {
                    bestFace = face;
                }
            }

            Rect bounds = bestFace.getBoundingBox();
            return new DetectionResult(cropFace(bitmap, bounds), bounds);

        } catch (Exception e) {
            android.util.Log.w("FaceDetectorHelper", "Face detection failed, using original image", e);
            return new DetectionResult(bitmap, null);
        }
    }

    /**
     * Croppe le visage avec une marge de 20% autour du bounding box.
     */
    private Bitmap cropFace(Bitmap bitmap, Rect boundingBox) {
        int imgWidth = bitmap.getWidth();
        int imgHeight = bitmap.getHeight();

        // Ajouter une marge
        int marginX = (int) (boundingBox.width() * MARGIN_RATIO);
        int marginY = (int) (boundingBox.height() * MARGIN_RATIO);

        // Calculer les coordonnees avec marge, clampees aux bords de l'image
        int left = Math.max(0, boundingBox.left - marginX);
        int top = Math.max(0, boundingBox.top - marginY);
        int right = Math.min(imgWidth, boundingBox.right + marginX);
        int bottom = Math.min(imgHeight, boundingBox.bottom + marginY);

        int cropWidth = right - left;
        int cropHeight = bottom - top;

        if (cropWidth <= 0 || cropHeight <= 0) {
            return bitmap;
        }

        return Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight);
    }

    public void close() {
        detector.close();
    }
}
