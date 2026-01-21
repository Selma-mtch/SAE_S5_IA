package com.example.ia_ethnie.ui.info;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.databinding.ActivityModelInfoBinding;
import com.example.ia_ethnie.ml.FaceAnalyzer;

public class ModelInfoActivity extends AppCompatActivity {
    private ActivityModelInfoBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityModelInfoBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupUI();
        setupListeners();
    }

    private void setupUI() {
        // Informations générales
        binding.tvProjectTitle.setText("Projet SAE S5 - IA Reconnaissance Faciale");

        binding.tvProjectDesc.setText(
                "Application mobile Android intelligente pour la prédiction de l'âge, " +
                "du genre et de l'ethnicité à partir d'images de visages."
        );

        // Dataset
        binding.tvDataset.setText(
                "Dataset: UTKFace\n" +
                "- 20,000+ images de visages\n" +
                "- Labels: âge, genre, ethnicité\n" +
                "- Format: [age]_[gender]_[race]_[date].jpg"
        );

        // Modèles
        binding.tvModels.setText(
                "Stratégies de modélisation:\n\n" +
                "1. Modèles spécialisés\n" +
                "   - Un modèle pour l'âge (régression)\n" +
                "   - Un modèle pour le genre (classification binaire)\n" +
                "   - Un modèle pour l'ethnicité (5 classes)\n\n" +
                "2. Modèle multi-tâches\n" +
                "   - Architecture: ResNet + SE-Net\n" +
                "   - Entrée: 128x128 niveaux de gris\n" +
                "   - 3 sorties simultanées\n\n" +
                "3. Transfert learning\n" +
                "   - Base: MobileNetV2 / EfficientNetB0\n" +
                "   - Fine-tuning sur UTKFace"
        );

        // Métriques
        binding.tvMetrics.setText(
                "Métriques d'évaluation:\n\n" +
                "Classification:\n" +
                "- Accuracy (précision globale)\n" +
                "- AUC (Area Under Curve)\n" +
                "- AP (Average Precision)\n\n" +
                "Régression (âge):\n" +
                "- MAE (Mean Absolute Error)\n" +
                "- MSE (Mean Squared Error)\n" +
                "- R² (Coefficient de détermination)\n\n" +
                "Clustering:\n" +
                "- ARI (Adjusted Rand Index)\n" +
                "- NMI (Normalized Mutual Information)"
        );

        // Technologies
        binding.tvTech.setText(
                "Technologies:\n\n" +
                "- TensorFlow / Keras (entraînement)\n" +
                "- TensorFlow Lite (déploiement mobile)\n" +
                "- Android Studio / Java\n" +
                "- CameraX (capture photo)\n" +
                "- Room Database (SQLite)\n" +
                "- Material Design"
        );

        // Classes ethnicité
        StringBuilder ethnicities = new StringBuilder("Classes d'ethnicité:\n");
        for (int i = 0; i < FaceAnalyzer.ETHNICITY_LABELS.length; i++) {
            ethnicities.append("- ").append(FaceAnalyzer.ETHNICITY_LABELS[i]).append("\n");
        }
        binding.tvEthnicities.setText(ethnicities.toString());
    }

    private void setupListeners() {
        binding.btnBack.setOnClickListener(v -> finish());
    }
}
