package com.example.ia_ethnie.ui.main;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.ia_ethnie.R;
import com.example.ia_ethnie.databinding.ActivityMainBinding;
import com.example.ia_ethnie.ml.FaceAnalyzer;
import com.example.ia_ethnie.ui.auth.LoginActivity;
import com.example.ia_ethnie.ui.camera.CameraActivity;
import com.example.ia_ethnie.ui.history.HistoryActivity;
import com.example.ia_ethnie.ui.info.ModelInfoActivity;
import com.example.ia_ethnie.ui.result.ResultActivity;
import com.example.ia_ethnie.utils.SessionManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private SessionManager sessionManager;
    private static final int CAMERA_PERMISSION_CODE = 100;
    private Uri currentImageUri;

    private final ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    if (imageUri != null) {
                        processImage(imageUri);
                    }
                }
            }
    );

    private final ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK) {
                    String imagePath = result.getData() != null ?
                            result.getData().getStringExtra("image_path") : null;
                    if (imagePath != null) {
                        processImage(Uri.fromFile(new File(imagePath)));
                    }
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        sessionManager = new SessionManager(this);

        // Vérifier si connecté
        if (!sessionManager.isLoggedIn()) {
            navigateToLogin();
            return;
        }

        setupUI();
        setupListeners();
    }

    private void setupUI() {
        setSupportActionBar(binding.toolbar);
        binding.tvUsername.setText(sessionManager.getUsername());
    }

    private void setupListeners() {
        binding.cardCamera.setOnClickListener(v -> {
            if (checkCameraPermission()) {
                openCamera();
            }
        });

        binding.cardGallery.setOnClickListener(v -> openGallery());

        binding.cardRealtime.setOnClickListener(v -> {
            if (checkCameraPermission()) {
                openRealtimeCamera();
            }
        });

        binding.cardHistory.setOnClickListener(v -> {
            startActivity(new Intent(this, HistoryActivity.class));
        });

        binding.cardModelInfo.setOnClickListener(v -> {
            startActivity(new Intent(this, ModelInfoActivity.class));
        });

        binding.cardSettings.setOnClickListener(v -> showModelTypeDialog());
    }

    private boolean checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Permission caméra requise", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void openCamera() {
        Intent intent = new Intent(this, CameraActivity.class);
        intent.putExtra("mode", "capture");
        cameraLauncher.launch(intent);
    }

    private void openRealtimeCamera() {
        Intent intent = new Intent(this, CameraActivity.class);
        intent.putExtra("mode", "realtime");
        startActivity(intent);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        galleryLauncher.launch(intent);
    }

    private void processImage(Uri imageUri) {
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
            String imagePath = saveImageToCache(bitmap);

            Intent intent = new Intent(this, ResultActivity.class);
            intent.putExtra("image_path", imagePath);
            startActivity(intent);
        } catch (IOException e) {
            Toast.makeText(this, "Erreur lors du chargement de l'image", Toast.LENGTH_SHORT).show();
        }
    }

    private String saveImageToCache(Bitmap bitmap) throws IOException {
        File cacheDir = new File(getCacheDir(), "images");
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
        File imageFile = new File(cacheDir, "image_" + System.currentTimeMillis() + ".jpg");
        try (FileOutputStream fos = new FileOutputStream(imageFile)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
        }
        return imageFile.getAbsolutePath();
    }

    private void showModelTypeDialog() {
        String[] options = {"Modèles spécialisés", "Modèle multi-tâches", "Transfert learning"};
        int currentSelection = getSharedPreferences("settings", MODE_PRIVATE)
                .getInt("model_type", 1);

        new AlertDialog.Builder(this)
                .setTitle("Type de modèle")
                .setSingleChoiceItems(options, currentSelection, (dialog, which) -> {
                    getSharedPreferences("settings", MODE_PRIVATE)
                            .edit()
                            .putInt("model_type", which)
                            .apply();
                    Toast.makeText(this, "Modèle: " + options[which], Toast.LENGTH_SHORT).show();
                    dialog.dismiss();
                })
                .setNegativeButton("Annuler", null)
                .show();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.action_logout) {
            logout();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void logout() {
        new AlertDialog.Builder(this)
                .setTitle("Déconnexion")
                .setMessage("Voulez-vous vraiment vous déconnecter?")
                .setPositiveButton("Oui", (dialog, which) -> {
                    sessionManager.logout();
                    navigateToLogin();
                })
                .setNegativeButton("Non", null)
                .show();
    }

    private void navigateToLogin() {
        Intent intent = new Intent(this, LoginActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
}
