/// Port of pipeline/utilities.py Config — the constants the ACTIVE-state rally
/// pipeline depends on.
class EngineConfig {
  static const int analysisWidth = 960;
  static const int analysisHeight = 540;

  // Rectangular model input for the whole-frame yolo26n/ball_best detectors
  // (spikes/export_mobile_models.py): the 960x540 analysis frame letterboxed
  // to the stride-32-rounded 960x544, instead of padding all the way to a
  // square 960x960 — ~1.77x fewer pixels per inference, same weights/accuracy
  // (see spikes/fixtures/rect_letterbox_meta.json for the derivation).
  static const int modelWidth = 960;
  static const int modelHeight = 544;
  static const int modelPadTop = 2; // r=1.0, dw=0, dh=2 top / 2 bottom

  // Square far-crop model input (unchanged from the original spike): used
  // ONLY by DeadTimeEngine's native far-region crop (FixedFarCropSource),
  // which pads an arbitrary-aspect crop into a square tensor. Keeping this
  // geometry unchanged preserves the tuned crop-based far-serve detection
  // (CutterConfig.farCropTopExtendFrac) independent of the rect optimization.
  static const int imgsz = 960; // model input (letterboxed square)

  // Output grid lengths (N in the raw [1,5,N] ball layout) for the two ball
  // model variants — fixed since both graphs export at a static input shape.
  static const int ballRectOutputN = 10710; // ball_best.onnx (960x544)
  static const int ballSquareOutputN = 18900; // ball_best_far_crop.onnx (960x960)
  static const int playerOutputRows = 300; // end2end max detections, both models

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

  // Fixed far-region crop for fballs (FixedFarCropSource): the bounding box of
  // the far HALF of the court (net line → far baseline) in pixel space, from
  // the calibrated corners via homography, with the TOP edge extended upward
  // by this fraction of the box height to reach above the far baseline.
  // Replaces the Python player-anchored crop for the mobile port — no
  // far-player-box dependency, computed once at calibration.
  //
  // The far half is heavily foreshortened (a ~17px sliver on folder 68), so
  // the toss/contact — where the far serve's ball actually is — sits WELL
  // above it.  Validated by re-running stage 2 on folder-68 telemetry with
  // fballs filtered to the candidate crop: 0.5 kept only 9/22 far serves, but
  // 3.0 (top ~52px above the far baseline) recovers the full 16/22 baseline.
  static const double farCropTopExtendFrac = 3.0;

  static const int playerClassIndex = 0;
  static const int ballClassIndex   = 0;
}
