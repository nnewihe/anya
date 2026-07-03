import 'package:flutter/material.dart';

/// Anya Tennis brand palette: black court, yellow ball, sky-blue line calls.
class AppColors {
  AppColors._();

  static const Color black = Color(0xFF000000);
  static const Color surface = Color(0xFF141412);
  static const Color surfaceAlt = Color(0xFF1F1F1B);
  static const Color yellow = Color(0xFFE8FF3D);
  static const Color skyBlue = Color(0xFF49C5F1);
  static const Color outline = Color(0xFF3A3A36);
  static const Color textDim = Color(0xFFA0A099);
}

class AppTheme {
  AppTheme._();

  static ThemeData get dark {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.black,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.yellow,
        onPrimary: AppColors.black,
        secondary: AppColors.skyBlue,
        onSecondary: AppColors.black,
        surface: AppColors.surface,
        onSurface: Colors.white,
        error: Color(0xFFFF6B6B),
      ),
    );

    return base.copyWith(
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.black,
        foregroundColor: Colors.white,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontFamily: 'Helvetica Neue',
          fontSize: 18,
          fontWeight: FontWeight.w500,
          letterSpacing: 4,
          color: Colors.white,
        ),
      ),
      textTheme: base.textTheme.apply(
        bodyColor: Colors.white,
        displayColor: Colors.white,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.yellow,
          foregroundColor: AppColors.black,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(2)),
          ),
          textStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            letterSpacing: 1,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: Colors.white,
          side: const BorderSide(color: AppColors.outline),
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(2)),
          ),
        ),
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: AppColors.yellow,
        linearTrackColor: AppColors.surfaceAlt,
      ),
      dividerTheme: const DividerThemeData(color: AppColors.outline),
      chipTheme: base.chipTheme.copyWith(
        backgroundColor: AppColors.surfaceAlt,
        labelStyle: const TextStyle(color: AppColors.skyBlue, fontSize: 12),
        side: const BorderSide(color: AppColors.outline),
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(2)),
        ),
      ),
      listTileTheme: const ListTileThemeData(
        iconColor: AppColors.skyBlue,
        textColor: Colors.white,
      ),
      snackBarTheme: const SnackBarThemeData(
        backgroundColor: AppColors.surfaceAlt,
        contentTextStyle: TextStyle(color: Colors.white),
      ),
    );
  }
}
