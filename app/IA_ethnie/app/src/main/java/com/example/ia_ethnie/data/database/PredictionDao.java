package com.example.ia_ethnie.data.database;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.ia_ethnie.data.model.Prediction;

import java.util.List;

@Dao
public interface PredictionDao {
    @Insert
    long insert(Prediction prediction);

    @Delete
    void delete(Prediction prediction);

    @Query("SELECT * FROM predictions WHERE userId = :userId ORDER BY createdAt DESC")
    List<Prediction> findByUserId(int userId);

    @Query("SELECT * FROM predictions WHERE id = :id LIMIT 1")
    Prediction findById(int id);

    @Query("SELECT * FROM predictions ORDER BY createdAt DESC LIMIT :limit")
    List<Prediction> getRecent(int limit);

    @Query("SELECT * FROM predictions WHERE userId = :userId ORDER BY createdAt DESC LIMIT :limit")
    List<Prediction> getRecentByUser(int userId, int limit);

    @Query("DELETE FROM predictions WHERE userId = :userId")
    void deleteAllByUser(int userId);

    @Query("SELECT COUNT(*) FROM predictions WHERE userId = :userId")
    int countByUser(int userId);
}
