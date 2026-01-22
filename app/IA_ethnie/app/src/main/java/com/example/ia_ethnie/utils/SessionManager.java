package com.example.ia_ethnie.utils;

import android.content.Context;
import android.content.SharedPreferences;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;

public class SessionManager {
    private static final String PREF_NAME = "IAEthnieSession";
    private static final String KEY_USERNAME = "username";

    private final SharedPreferences prefs;
    private final SharedPreferences.Editor editor;
    private final FirebaseAuth auth;

    public SessionManager(Context context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        editor = prefs.edit();
        auth = FirebaseAuth.getInstance();
    }

    public void saveUsername(String username) {
        editor.putString(KEY_USERNAME, username);
        editor.apply();
    }

    public boolean isLoggedIn() {
        return auth.getCurrentUser() != null;
    }

    public String getUserId() {
        FirebaseUser user = auth.getCurrentUser();
        return user != null ? user.getUid() : null;
    }

    public String getUsername() {
        return prefs.getString(KEY_USERNAME, "Utilisateur");
    }

    public String getEmail() {
        FirebaseUser user = auth.getCurrentUser();
        return user != null ? user.getEmail() : "";
    }

    public void logout() {
        auth.signOut();
        editor.clear();
        editor.apply();
    }
}
