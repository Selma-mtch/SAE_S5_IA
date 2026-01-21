package com.example.ia_ethnie.ui.history;

import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.ia_ethnie.R;
import com.example.ia_ethnie.data.database.AppDatabase;
import com.example.ia_ethnie.data.model.Prediction;
import com.example.ia_ethnie.databinding.ActivityHistoryBinding;
import com.example.ia_ethnie.utils.SessionManager;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class HistoryActivity extends AppCompatActivity {
    private ActivityHistoryBinding binding;
    private AppDatabase database;
    private SessionManager sessionManager;
    private HistoryAdapter adapter;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private List<Prediction> predictions = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityHistoryBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        database = AppDatabase.getInstance(this);
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
        executor.execute(() -> {
            List<Prediction> loadedPredictions = database.predictionDao()
                    .findByUserId(sessionManager.getUserId());

            runOnUiThread(() -> {
                predictions = loadedPredictions;
                adapter.notifyDataSetChanged();

                if (predictions.isEmpty()) {
                    binding.tvEmpty.setVisibility(View.VISIBLE);
                    binding.recyclerView.setVisibility(View.GONE);
                } else {
                    binding.tvEmpty.setVisibility(View.GONE);
                    binding.recyclerView.setVisibility(View.VISIBLE);
                }
            });
        });
    }

    private void clearHistory() {
        executor.execute(() -> {
            database.predictionDao().deleteAllByUser(sessionManager.getUserId());
            runOnUiThread(() -> {
                predictions.clear();
                adapter.notifyDataSetChanged();
                binding.tvEmpty.setVisibility(View.VISIBLE);
                binding.recyclerView.setVisibility(View.GONE);
            });
        });
    }

    private void deletePrediction(Prediction prediction, int position) {
        executor.execute(() -> {
            database.predictionDao().delete(prediction);
            runOnUiThread(() -> {
                predictions.remove(position);
                adapter.notifyItemRemoved(position);
                if (predictions.isEmpty()) {
                    binding.tvEmpty.setVisibility(View.VISIBLE);
                    binding.recyclerView.setVisibility(View.GONE);
                }
            });
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
            Prediction prediction = predictions.get(position);

            // Image
            File imageFile = new File(prediction.getImagePath());
            if (imageFile.exists()) {
                holder.ivImage.setImageBitmap(BitmapFactory.decodeFile(prediction.getImagePath()));
            }

            // Infos
            holder.tvAge.setText("Âge: " + prediction.getPredictedAge());
            holder.tvGender.setText("Genre: " + prediction.getPredictedGender());
            holder.tvEthnicity.setText("Ethnicité: " + prediction.getPredictedEthnicity());

            // Date
            SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.FRENCH);
            holder.tvDate.setText(sdf.format(new Date(prediction.getCreatedAt())));

            // Modèle
            holder.tvModel.setText("Modèle: " + prediction.getModelType());

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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
    }
}
