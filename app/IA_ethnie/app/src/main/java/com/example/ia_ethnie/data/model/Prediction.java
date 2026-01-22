package com.example.ia_ethnie.data.model;

import androidx.room.Entity;
import androidx.room.ForeignKey;
import androidx.room.Index;
import androidx.room.PrimaryKey;

@Entity(tableName = "predictions",
        foreignKeys = @ForeignKey(
                entity = User.class,
                parentColumns = "id",
                childColumns = "userId",
                onDelete = ForeignKey.CASCADE
        ),
        indices = @Index("userId"))
public class Prediction {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private int userId;
    private String imagePath;
    private int predictedAge;
    private String predictedGender;
    private String predictedEthnicity;
    private float ageConfidence;
    private float genderConfidence;
    private float ethnicityConfidence;
    private String modelType;
    private long createdAt;

    public Prediction(int userId, String imagePath, int predictedAge, String predictedGender,
                      String predictedEthnicity, float ageConfidence, float genderConfidence,
                      float ethnicityConfidence, String modelType) {
        this.userId = userId;
        this.imagePath = imagePath;
        this.predictedAge = predictedAge;
        this.predictedGender = predictedGender;
        this.predictedEthnicity = predictedEthnicity;
        this.ageConfidence = ageConfidence;
        this.genderConfidence = genderConfidence;
        this.ethnicityConfidence = ethnicityConfidence;
        this.modelType = modelType;
        this.createdAt = System.currentTimeMillis();
    }

    // Getters et Setters
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    public int getUserId() { return userId; }
    public void setUserId(int userId) { this.userId = userId; }

    public String getImagePath() { return imagePath; }
    public void setImagePath(String imagePath) { this.imagePath = imagePath; }

    public int getPredictedAge() { return predictedAge; }
    public void setPredictedAge(int predictedAge) { this.predictedAge = predictedAge; }

    public String getPredictedGender() { return predictedGender; }
    public void setPredictedGender(String predictedGender) { this.predictedGender = predictedGender; }

    public String getPredictedEthnicity() { return predictedEthnicity; }
    public void setPredictedEthnicity(String predictedEthnicity) { this.predictedEthnicity = predictedEthnicity; }

    public float getAgeConfidence() { return ageConfidence; }
    public void setAgeConfidence(float ageConfidence) { this.ageConfidence = ageConfidence; }

    public float getGenderConfidence() { return genderConfidence; }
    public void setGenderConfidence(float genderConfidence) { this.genderConfidence = genderConfidence; }

    public float getEthnicityConfidence() { return ethnicityConfidence; }
    public void setEthnicityConfidence(float ethnicityConfidence) { this.ethnicityConfidence = ethnicityConfidence; }

    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }

    public long getCreatedAt() { return createdAt; }
    public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }
}
