/// Port of pipeline/utilities.py Config — the constants the ACTIVE-state rally
/// pipeline depends on.
class EngineConfig {
  static const int analysisWidth = 960;
  static const int analysisHeight = 540;
  static const int imgsz = 960; // model input (letterboxed square)

  static const double playerConf = 0.5;
  static const double activeBallConf = 0.10; // ACTIVE_BALL_CONF
  static const double exclusionScanConf = 0.04;

  static const double courtWidthFt = 27.0;
  static const double courtLengthFt = 78.0;
  static const double nearPlayerXPadFt = 3.0;

  static const int playerClassIndex = 0;
  static const int ballClassIndex = 0;

  // Exclusion-zone scan (create_auto_exclusion_zones defaults).
  static const int exclusionScanFrames = 50;
  static const double exclusionEps = 12;
  static const int exclusionMinSamples = 15;
}
