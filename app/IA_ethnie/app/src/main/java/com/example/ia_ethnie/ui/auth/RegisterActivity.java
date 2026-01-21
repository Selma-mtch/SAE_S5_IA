package com.example.ia_ethnie.ui.auth;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ia_ethnie.data.database.AppDatabase;
import com.example.ia_ethnie.data.model.User;
import com.example.ia_ethnie.databinding.ActivityRegisterBinding;
import com.example.ia_ethnie.ui.main.MainActivity;
import com.example.ia_ethnie.utils.SessionManager;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RegisterActivity extends AppCompatActivity {
    private ActivityRegisterBinding binding;
    private AppDatabase database;
    private SessionManager sessionManager;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityRegisterBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        database = AppDatabase.getInstance(this);
        sessionManager = new SessionManager(this);

        setupListeners();
    }

    private void setupListeners() {
        binding.btnRegister.setOnClickListener(v -> attemptRegister());

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

        executor.execute(() -> {
            // Vérifier si l'email existe déjà
            if (database.userDao().emailExists(email)) {
                runOnUiThread(() -> {
                    binding.btnRegister.setEnabled(true);
                    Toast.makeText(this, "Cet email est déjà utilisé", Toast.LENGTH_SHORT).show();
                });
                return;
            }

            // Vérifier si le nom d'utilisateur existe déjà
            if (database.userDao().usernameExists(username)) {
                runOnUiThread(() -> {
                    binding.btnRegister.setEnabled(true);
                    Toast.makeText(this, "Ce nom d'utilisateur est déjà pris", Toast.LENGTH_SHORT).show();
                });
                return;
            }

            // Créer l'utilisateur
            User newUser = new User(username, email, password);
            long userId = database.userDao().insert(newUser);

            runOnUiThread(() -> {
                binding.btnRegister.setEnabled(true);
                if (userId > 0) {
                    sessionManager.createSession((int) userId, username, email);
                    Toast.makeText(this, "Compte créé avec succès!", Toast.LENGTH_SHORT).show();
                    navigateToMain();
                } else {
                    Toast.makeText(this, "Erreur lors de la création du compte", Toast.LENGTH_SHORT).show();
                }
            });
        });
    }

    private void navigateToMain() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }
}
