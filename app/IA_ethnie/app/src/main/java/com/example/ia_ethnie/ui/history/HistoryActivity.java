package com.example.ia_ethnie.ui.history;

import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.ia_ethnie.R;
import com.example.ia_ethnie.databinding.ActivityHistoryBinding;
import com.example.ia_ethnie.utils.SessionManager;
import com.google.firebase.firestore.DocumentSnapshot;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.Query;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class HistoryActivity extends AppCompatActivity {
    private ActivityHistoryBinding binding;
    private FirebaseFirestore db;
    private SessionManager sessionManager;
    private HistoryAdapter adapter;
    private List<PredictionItem> predictions = new ArrayList<>();

    // Classe interne pour stocker les données de prédiction
    private static class PredictionItem {
        String documentId;
        String localImagePath;
        int age;
        String gender;
        String ethnicity;
        String modelType;
        long createdAt;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityHistoryBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        db = FirebaseFirestore.getInstance();
        sessionManager = new SessionManager(this);

        setupRecyclerView();
        setupListeners();
        loadHistory();
    }

    private void setupRecyclerView() {
        adapter = new HistoryAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);
    }

    private void setupListeners() {
        binding.btnBack.setOnClickListener(v -> finish());

        binding.btnClearAll.setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setTitle("Supprimer l'historique")
                    .setMessage("Voulez-vous supprimer tout l'historique?")
                    .setPositiveButton("Oui", (dialog, which) -> clearHistory())
                    .setNegativeButton("Non", null)
                    .show();
        });
    }

    private void loadHistory() {
        String userId = sessionManager.getUserId();
        if (userId == null) {
            showEmptyState();
            return;
        }

        db.collection("predictions")
                .whereEqualTo("userId", userId)
                .orderBy("createdAt", Query.Direction.DESCENDING)
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    predictions.clear();

                    for (DocumentSnapshot doc : querySnapshot.getDocuments()) {
                        PredictionItem item = new PredictionItem();
                        item.documentId = doc.getId();
                        item.localImagePath = doc.getString("localImagePath");

                        // Gérer les types numériques (Long vs Integer)
                        Long ageLong = doc.getLong("age");
                        item.age = ageLong != null ? ageLong.intValue() : 0;

                        item.gender = doc.getString("gender");
                        item.ethnicity = doc.getString("ethnicity");
                        item.modelType = doc.getString("modelType");

                        Long createdAtLong = doc.getLong("createdAt");
                        item.createdAt = createdAtLong != null ? createdAtLong : 0;

                        predictions.add(item);
                    }

                    adapter.notifyDataSetChanged();

                    if (predictions.isEmpty()) {
                        showEmptyState();
                    } else {
                        binding.tvEmpty.setVisibility(View.GONE);
                        binding.recyclerView.setVisibility(View.VISIBLE);
                    }
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(this, "Erreur de chargement", Toast.LENGTH_SHORT).show();
                    showEmptyState();
                });
    }

    private void showEmptyState() {
        binding.tvEmpty.setVisibility(View.VISIBLE);
        binding.recyclerView.setVisibility(View.GONE);
    }

    private void clearHistory() {
        String userId = sessionManager.getUserId();
        if (userId == null) return;

        // Supprimer tous les documents de l'utilisateur
        db.collection("predictions")
                .whereEqualTo("userId", userId)
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    for (DocumentSnapshot doc : querySnapshot.getDocuments()) {
                        doc.getReference().delete();
                    }
                    predictions.clear();
                    adapter.notifyDataSetChanged();
                    showEmptyState();
                    Toast.makeText(this, "Historique supprimé", Toast.LENGTH_SHORT).show();
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(this, "Erreur lors de la suppression", Toast.LENGTH_SHORT).show();
                });
    }

    private void deletePrediction(PredictionItem prediction, int position) {
        db.collection("predictions").document(prediction.documentId)
                .delete()
                .addOnSuccessListener(aVoid -> {
                    predictions.remove(position);
                    adapter.notifyItemRemoved(position);
                    if (predictions.isEmpty()) {
                        showEmptyState();
                    }
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(this, "Erreur lors de la suppression", Toast.LENGTH_SHORT).show();
                });
    }

    private class HistoryAdapter extends RecyclerView.Adapter<HistoryAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_history, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            PredictionItem prediction = predictions.get(position);

            // Image locale
            if (prediction.localImagePath != null) {
                File imageFile = new File(prediction.localImagePath);
                if (imageFile.exists()) {
                    holder.ivImage.setImageBitmap(BitmapFactory.decodeFile(prediction.localImagePath));
                } else {
                    holder.ivImage.setImageResource(R.drawable.ic_face_scan);
                }
            } else {
                holder.ivImage.setImageResource(R.drawable.ic_face_scan);
            }

            // Infos
            holder.tvAge.setText(prediction.age + " ans");
            holder.tvGender.setText(prediction.gender != null ? prediction.gender : "--");
            holder.tvEthnicity.setText(prediction.ethnicity != null ? prediction.ethnicity : "--");

            // Date
            SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.FRENCH);
            holder.tvDate.setText(sdf.format(new Date(prediction.createdAt)));

            // Modèle
            holder.tvModel.setText(prediction.modelType != null ? prediction.modelType : "N/A");

            // Suppression
            holder.btnDelete.setOnClickListener(v -> {
                new AlertDialog.Builder(HistoryActivity.this)
                        .setTitle("Supprimer")
                        .setMessage("Supprimer cette prédiction?")
                        .setPositiveButton("Oui", (d, w) ->
                                deletePrediction(prediction, holder.getAdapterPosition()))
                        .setNegativeButton("Non", null)
                        .show();
            });
        }

        @Override
        public int getItemCount() {
            return predictions.size();
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            ImageView ivImage;
            TextView tvAge, tvGender, tvEthnicity, tvDate, tvModel;
            View btnDelete;

            ViewHolder(View view) {
                super(view);
                ivImage = view.findViewById(R.id.ivImage);
                tvAge = view.findViewById(R.id.tvAge);
                tvGender = view.findViewById(R.id.tvGender);
                tvEthnicity = view.findViewById(R.id.tvEthnicity);
                tvDate = view.findViewById(R.id.tvDate);
                tvModel = view.findViewById(R.id.tvModel);
                btnDelete = view.findViewById(R.id.btnDelete);
            }
        }
    }
}
