package com.example.ia_ethnie.ui.camera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

/**
 * Vue transparente qui dessine un rectangle autour du visage detecte.
 */
public class FaceOverlayView extends View {

    private Rect faceBounds;
    private final Paint boxPaint;
    private final Paint textBgPaint;
    private final Paint textPaint;
    private String label;
    private int sourceWidth = 1;
    private int sourceHeight = 1;
    private boolean isFrontCamera = false;

    public FaceOverlayView(Context context) {
        this(context, null);
    }

    public FaceOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

        boxPaint = new Paint();
        boxPaint.setColor(Color.parseColor("#4CAF50"));
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        boxPaint.setAntiAlias(true);

        textBgPaint = new Paint();
        textBgPaint.setColor(Color.parseColor("#CC1A1A1A"));
        textBgPaint.setStyle(Paint.Style.FILL);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(32f);
        textPaint.setAntiAlias(true);
    }

    public void setFaceBounds(Rect bounds, int srcWidth, int srcHeight, boolean frontCamera, String faceLabel) {
        this.faceBounds = bounds;
        this.sourceWidth = srcWidth;
        this.sourceHeight = srcHeight;
        this.isFrontCamera = frontCamera;
        this.label = faceLabel;
        postInvalidate();
    }

    public void clearBounds() {
        this.faceBounds = null;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (faceBounds == null) return;

        int viewW = getWidth();
        int viewH = getHeight();
        if (viewW == 0 || viewH == 0 || sourceWidth == 0 || sourceHeight == 0) return;

        // Adapter les coordonnees de l'image source a la taille de la vue
        float scaleX = (float) viewW / sourceWidth;
        float scaleY = (float) viewH / sourceHeight;

        float left = faceBounds.left * scaleX;
        float top = faceBounds.top * scaleY;
        float right = faceBounds.right * scaleX;
        float bottom = faceBounds.bottom * scaleY;

        // Miroir horizontal pour camera frontale
        if (isFrontCamera) {
            float tmpLeft = viewW - right;
            right = viewW - left;
            left = tmpLeft;
        }

        // Rectangle autour du visage
        canvas.drawRect(left, top, right, bottom, boxPaint);

        // Label au-dessus du rectangle
        if (label != null && !label.isEmpty()) {
            float textWidth = textPaint.measureText(label);
            float textHeight = 40f;
            canvas.drawRect(left, top - textHeight - 8, left + textWidth + 16, top, textBgPaint);
            canvas.drawText(label, left + 8, top - 12, textPaint);
        }
    }
}
