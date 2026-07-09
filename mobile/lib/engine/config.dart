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

/// Stage-1 constants for the DEAD-TIME CUTTER's telemetry extractor
/// (match_telemetry.dart) — mirrors pipeline/match_telemetry.py ExtractorConfig
/// plus the toss/far values from utilities.Config.  Kept separate from
/// EngineConfig, which drives the rally engine: the two pipelines share the
/// same two ONNX models but use different thresholds (notably playerConf is
/// LOWER here — the far player needs the low floor).
class CutterConfig {
  static const double playerConf    = 0.2;    // far player needs the low floor
  static const double nearMinConf   = 0.5;    // near-player candidates clear this
  static const double ballConf      = 0.10;   // whole-court ball (ACTIVE_BALL_CONF)
  static const double tossConf      = 0.10;   // toss-ROI ball (TOSS_BALL_CONF)
  static const int    tossBallImgsz = 320;    // toss ROI is small — 320 input
  static const int    farBallImgsz  = 480;    // native far crop ~340x460

  // Court geometry (feet) — dst rectangle for the homography.
  static const double courtWidthFt     = 27.0;
  static const double courtLengthFt    = 78.0;
  static const double nearPlayerXPadFt = 3.0; // homography-tolerance padding
  static const double farPlayerXPadFt  = 3.0;

  // Far-player robustness (match ExtractorConfig).
  static const double farBoxHoldS     = 0.7;  // hold last box through gaps
  static const double farWorldSmoothS = 0.3;  // smooth homography-amplified jitter

  // Far native-crop gate: only crop when the far player is within this signed
  // distance band of the far baseline (positive = behind it).  Deliberately
  // loose — homography amplifies far-court pixel noise into tens of feet.
  static const double farGateMinFt = -15.0;
  static const double farGateMaxFt = 40.0;

  static const int playerClassIndex = 0;
  static const int ballClassIndex   = 0;
}
