package com.example.ia_ethnie.ui.auth;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.databinding.ActivityRegisterBinding;
import com.example.ia_ethnie.ui.main.MainActivity;
import com.example.ia_ethnie.utils.SessionManager;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.firestore.FirebaseFirestore;

import java.util.HashMap;
import java.util.Map;

public class RegisterActivity extends AppCompatActivity {
    private ActivityRegisterBinding binding;
    private FirebaseAuth auth;
    private FirebaseFirestore db;
    private SessionManager sessionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityRegisterBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        auth = FirebaseAuth.getInstance();
        db = FirebaseFirestore.getInstance();
        sessionManager = new SessionManager(this);

        setupListeners();
    }

    private void setupListeners() {
        binding.btnRegister.setOnClickListener(v -> attemptRegister());

        binding.btnBack.setOnClickListener(v -> finish());

        binding.tvLogin.setOnClickListener(v -> finish());
    }

    private void attemptRegister() {
        String username = binding.etUsername.getText().toString().trim();
        String email = binding.etEmail.getText().toString().trim();
        String password = binding.etPassword.getText().toString().trim();
        String confirmPassword = binding.etConfirmPassword.getText().toString().trim();

        // Validation
        if (username.isEmpty() || email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Veuillez remplir tous les champs", Toast.LENGTH_SHORT).show();
            return;
        }

        if (username.length() < 3) {
            Toast.makeText(this, "Le nom d'utilisateur doit avoir au moins 3 caractères", Toast.LENGTH_SHORT).show();
            return;
        }

        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            Toast.makeText(this, "Email invalide", Toast.LENGTH_SHORT).show();
            return;
        }

        if (password.length() < 6) {
            Toast.makeText(this, "Le mot de passe doit avoir au moins 6 caractères", Toast.LENGTH_SHORT).show();
            return;
        }

        if (!password.equals(confirmPassword)) {
            Toast.makeText(this, "Les mots de passe ne correspondent pas", Toast.LENGTH_SHORT).show();
            return;
        }

        binding.btnRegister.setEnabled(false);

        // Créer l'utilisateur avec Firebase Auth
        auth.createUserWithEmailAndPassword(email, password)
                .addOnSuccessListener(authResult -> {
                    String uid = authResult.getUser().getUid();

                    // Sauvegarder les infos utilisateur dans Firestore
                    Map<String, Object> userData = new HashMap<>();
                    userData.put("username", username);
                    userData.put("email", email);
                    userData.put("createdAt", System.currentTimeMillis());

                    db.collection("users").document(uid).set(userData)
                            .addOnSuccessListener(aVoid -> {
                                sessionManager.saveUsername(username);
                                Toast.makeText(this, "Compte créé avec succès!", Toast.LENGTH_SHORT).show();
                                navigateToMain();
                            })
                            .addOnFailureListener(e -> {
                                // Même si Firestore échoue, le compte Auth est créé
                                sessionManager.saveUsername(username);
                                navigateToMain();
                            });
                })
                .addOnFailureListener(e -> {
                    binding.btnRegister.setEnabled(true);
                    String errorMessage = "Erreur lors de la création du compte";

                    if (e.getMessage() != null) {
                        if (e.getMessage().contains("email address is already")) {
                            errorMessage = "Cet email est déjà utilisé";
                        } else if (e.getMessage().contains("network")) {
                            errorMessage = "Erreur réseau. Vérifiez votre connexion.";
                        } else if (e.getMessage().contains("weak password")) {
                            errorMessage = "Mot de passe trop faible";
                        }
                    }

                    Toast.makeText(this, errorMessage, Toast.LENGTH_SHORT).show();
                });
    }

    private void navigateToMain() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
}
