package com.example.ia_ethnie.ui.result;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.data.database.AppDatabase;
import com.example.ia_ethnie.data.model.Prediction;
import com.example.ia_ethnie.databinding.ActivityResultBinding;
import com.example.ia_ethnie.ml.FaceAnalyzer;
import com.example.ia_ethnie.utils.SessionManager;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ResultActivity extends AppCompatActivity {
    private ActivityResultBinding binding;
    private FaceAnalyzer faceAnalyzer;
    private AppDatabase database;
    private SessionManager sessionManager;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private String imagePath;
    private FaceAnalyzer.PredictionResult currentResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityResultBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        database = AppDatabase.getInstance(this);
        sessionManager = new SessionManager(this);
        faceAnalyzer = new FaceAnalyzer(this);

        // Charger le type de modèle sélectionné
        int modelType = getSharedPreferences("settings", MODE_PRIVATE)
                .getInt("model_type", 1);
        faceAnalyzer.setModelType(FaceAnalyzer.ModelType.values()[modelType]);

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

    private void processImage() {
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        if (bitmap == null) {
            Toast.makeText(this, "Erreur lecture image", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        binding.ivFace.setImageBitmap(bitmap);

        executor.execute(() -> {
            currentResult = faceAnalyzer.analyze(bitmap);

            runOnUiThread(() -> displayResult(currentResult));
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

        executor.execute(() -> {
            Prediction prediction = new Prediction(
                    sessionManager.getUserId(),
                    imagePath,
                    currentResult.age,
                    currentResult.gender,
                    currentResult.ethnicity,
                    currentResult.ageConfidence,
                    currentResult.genderConfidence,
                    currentResult.ethnicityConfidence,
                    currentResult.modelType.name()
            );

            database.predictionDao().insert(prediction);

            runOnUiThread(() -> {
                Toast.makeText(this, "Résultat sauvegardé!", Toast.LENGTH_SHORT).show();
                binding.btnSave.setEnabled(false);
                binding.btnSave.setAlpha(0.5f);
            });
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
