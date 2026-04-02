package com.example.ia_ethnie.ui.result;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.databinding.ActivityResultBinding;
import com.example.ia_ethnie.ml.FaceAnalyzer;
import com.example.ia_ethnie.utils.SessionManager;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ResultActivity extends AppCompatActivity {
    private ActivityResultBinding binding;
    private FaceAnalyzer faceAnalyzer;
    private FirebaseFirestore db;
    private SessionManager sessionManager;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private String imagePath;
    private FaceAnalyzer.PredictionResult currentResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityResultBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        db = FirebaseFirestore.getInstance();
        sessionManager = new SessionManager(this);
        faceAnalyzer = new FaceAnalyzer(this);

        // Charger le type de modèle sélectionné (avec validation)
        int modelType = getSharedPreferences("settings", MODE_PRIVATE)
                .getInt("model_type", 1);
        FaceAnalyzer.ModelType[] types = FaceAnalyzer.ModelType.values();
        if (modelType < 0 || modelType >= types.length) {
            modelType = 1; // MULTI_TASK par défaut
        }
        faceAnalyzer.setModelType(types[modelType]);

        imagePath = getIntent().getStringExtra("image_path");
        if (imagePath == null) {
            Toast.makeText(this, "Erreur: pas d'image", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        setupListeners();
        processImage();
    }

    private void setupListeners() {
        binding.btnBack.setOnClickListener(v -> finish());

        binding.btnSave.setOnClickListener(v -> saveResult());

        binding.btnShare.setOnClickListener(v -> shareResult());
    }

    private Bitmap applyExifRotation(Bitmap bitmap, String path) {
        try {
            androidx.exifinterface.media.ExifInterface exif =
                    new androidx.exifinterface.media.ExifInterface(path);
            int orientation = exif.getAttributeInt(
                    androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                    androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL);

            int rotation = 0;
            switch (orientation) {
                case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90:
                    rotation = 90; break;
                case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180:
                    rotation = 180; break;
                case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270:
                    rotation = 270; break;
            }

            if (rotation == 0) return bitmap;

            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0,
                    bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap.recycle();
            return rotated;
        } catch (Exception e) {
            return bitmap;
        }
    }

    private void processImage() {
        executor.execute(() -> {
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap == null) {
                runOnUiThread(() -> {
                    Toast.makeText(this, "Erreur lecture image", Toast.LENGTH_SHORT).show();
                    finish();
                });
                return;
            }

            // Corriger la rotation EXIF (certains téléphones sauvegardent l'image à l'horizontale)
            final Bitmap finalBitmap = applyExifRotation(bitmap, imagePath);

            currentResult = faceAnalyzer.analyze(finalBitmap);

            runOnUiThread(() -> {
                // Afficher le visage croppe si detecte, sinon l'image originale
                if (currentResult.croppedFace != null) {
                    binding.ivFace.setImageBitmap(currentResult.croppedFace);
                } else {
                    binding.ivFace.setImageBitmap(finalBitmap);
                }
                displayResult(currentResult);
            });
        });
    }

    private void displayResult(FaceAnalyzer.PredictionResult result) {
        // Âge
        binding.tvAge.setText(result.age + " ans");
        binding.tvAgeConfidence.setText(String.format(Locale.FRENCH,
                "%.0f%%", result.ageConfidence * 100));
        binding.progressAge.setProgress((int) (result.ageConfidence * 100));

        // Genre
        binding.tvGender.setText(result.gender);
        binding.tvGenderConfidence.setText(String.format(Locale.FRENCH,
                "%.0f%%", result.genderConfidence * 100));
        binding.progressGender.setProgress((int) (result.genderConfidence * 100));

        // Ethnicité
        binding.tvEthnicity.setText(result.ethnicity);
        binding.tvEthnicityConfidence.setText(String.format(Locale.FRENCH,
                "%.0f%%", result.ethnicityConfidence * 100));
        binding.progressEthnicity.setProgress((int) (result.ethnicityConfidence * 100));

        // Type de modèle
        binding.tvModelType.setText(result.modelType.name());
    }

    private void saveResult() {
        if (currentResult == null) return;

        String userId = sessionManager.getUserId();
        if (userId == null) {
            Toast.makeText(this, "Erreur: utilisateur non connecté", Toast.LENGTH_SHORT).show();
            return;
        }

        binding.btnSave.setEnabled(false);

        // Créer le document pour Firestore
        Map<String, Object> prediction = new HashMap<>();
        prediction.put("userId", userId);
        prediction.put("localImagePath", imagePath);
        prediction.put("age", currentResult.age);
        prediction.put("gender", currentResult.gender);
        prediction.put("ethnicity", currentResult.ethnicity);
        prediction.put("ageConfidence", currentResult.ageConfidence);
        prediction.put("genderConfidence", currentResult.genderConfidence);
        prediction.put("ethnicityConfidence", currentResult.ethnicityConfidence);
        prediction.put("modelType", currentResult.modelType.name());
        prediction.put("createdAt", System.currentTimeMillis());

        db.collection("predictions").add(prediction)
                .addOnSuccessListener(documentReference -> {
                    Toast.makeText(this, "Résultat sauvegardé!", Toast.LENGTH_SHORT).show();
                    binding.btnSave.setAlpha(0.5f);
                })
                .addOnFailureListener(e -> {
                    binding.btnSave.setEnabled(true);
                    Toast.makeText(this, "Erreur lors de la sauvegarde", Toast.LENGTH_SHORT).show();
                });
    }

    private void shareResult() {
        if (currentResult == null) return;

        String shareText = String.format(Locale.FRENCH,
                "Analyse faciale IA\n" +
                "Âge: %d ans\n" +
                "Genre: %s\n" +
                "Ethnicité: %s\n" +
                "Modèle: %s",
                currentResult.age,
                currentResult.gender,
                currentResult.ethnicity,
                currentResult.modelType.name()
        );

        android.content.Intent shareIntent = new android.content.Intent();
        shareIntent.setAction(android.content.Intent.ACTION_SEND);
        shareIntent.putExtra(android.content.Intent.EXTRA_TEXT, shareText);
        shareIntent.setType("text/plain");
        startActivity(android.content.Intent.createChooser(shareIntent, "Partager via"));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
        faceAnalyzer.close();
    }
}
