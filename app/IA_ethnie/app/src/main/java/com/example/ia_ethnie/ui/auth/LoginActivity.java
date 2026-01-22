package com.example.ia_ethnie.ui.auth;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.databinding.ActivityLoginBinding;
import com.example.ia_ethnie.ui.main.MainActivity;
import com.example.ia_ethnie.utils.SessionManager;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.firestore.FirebaseFirestore;

public class LoginActivity extends AppCompatActivity {
    private ActivityLoginBinding binding;
    private FirebaseAuth auth;
    private FirebaseFirestore db;
    private SessionManager sessionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityLoginBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        auth = FirebaseAuth.getInstance();
        db = FirebaseFirestore.getInstance();
        sessionManager = new SessionManager(this);

        // Vérifier si déjà connecté
        if (sessionManager.isLoggedIn()) {
            navigateToMain();
            return;
        }

        setupListeners();
    }

    private void setupListeners() {
        binding.btnLogin.setOnClickListener(v -> attemptLogin());

        binding.tvRegister.setOnClickListener(v -> {
            startActivity(new Intent(this, RegisterActivity.class));
        });
    }

    private void attemptLogin() {
        String email = binding.etEmail.getText().toString().trim();
        String password = binding.etPassword.getText().toString().trim();

        if (email.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, "Veuillez remplir tous les champs", Toast.LENGTH_SHORT).show();
            return;
        }

        binding.btnLogin.setEnabled(false);

        auth.signInWithEmailAndPassword(email, password)
                .addOnSuccessListener(authResult -> {
                    // Récupérer le username depuis Firestore
                    String uid = authResult.getUser().getUid();
                    db.collection("users").document(uid).get()
                            .addOnSuccessListener(document -> {
                                String username = document.getString("username");
                                if (username != null) {
                                    sessionManager.saveUsername(username);
                                }
                                navigateToMain();
                            })
                            .addOnFailureListener(e -> {
                                // Même si on ne trouve pas le username, on continue
                                navigateToMain();
                            });
                })
                .addOnFailureListener(e -> {
                    binding.btnLogin.setEnabled(true);
                    String errorMessage = "Email ou mot de passe incorrect";
                    if (e.getMessage() != null && e.getMessage().contains("network")) {
                        errorMessage = "Erreur réseau. Vérifiez votre connexion.";
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
