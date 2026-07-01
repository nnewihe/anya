import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../engine/config.dart';
import '../engine/reference_frame.dart';
import 'analysis_screen.dart';

/// Court-corner calibration: the user taps the 4 court corners on a reference
/// frame, in the order [BL, BR, TR, TL] (matching the Python homography). The
/// tapped points are in analysis-frame (960×540) pixel space.
class CalibrationScreen extends StatefulWidget {
  final String videoPath;
  final String title;

  const CalibrationScreen({
    super.key,
    required this.videoPath,
    required this.title,
  });

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  static const _labels = [
    'near-left baseline corner',
    'near-right baseline corner',
    'far-right baseline corner',
    'far-left baseline corner',
  ];

  Uint8List? _frame;
  String? _error;
  final List<List<double>> _corners = []; // in 960×540 space

  @override
  void initState() {
    super.initState();
    _loadFrame();
  }

  Future<void> _loadFrame() async {
    try {
      final bytes = await extractReferenceFrame(widget.videoPath);
      if (mounted) setState(() => _frame = bytes);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  void _onTapUp(TapUpDetails d, Size renderSize) {
    if (_corners.length >= 4) return;
    // Map the tap (in the rendered image box) back to 960×540 space.
    final sx = EngineConfig.analysisWidth / renderSize.width;
    final sy = EngineConfig.analysisHeight / renderSize.height;
    setState(() {
      _corners.add([d.localPosition.dx * sx, d.localPosition.dy * sy]);
    });
  }

  void _reset() => setState(_corners.clear);

  void _analyze() {
    Navigator.of(context).pushReplacement(MaterialPageRoute(
      builder: (_) => AnalysisScreen(
        videoPath: widget.videoPath,
        corners: List.of(_corners),
        title: widget.title,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final done = _corners.length == 4;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mark the court'),
        actions: [
          if (_corners.isNotEmpty)
            IconButton(icon: const Icon(Icons.undo), onPressed: _reset),
        ],
      ),
      body: _error != null
          ? Center(child: Text('Could not load frame:\n$_error',
              textAlign: TextAlign.center))
          : _frame == null
              ? const Center(child: CircularProgressIndicator())
              : Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(
                        done
                            ? 'All 4 corners set — ready to analyze.'
                            : 'Tap the ${_labels[_corners.length]}  '
                                '(${_corners.length + 1}/4)',
                        style: Theme.of(context).textTheme.titleMedium,
                        textAlign: TextAlign.center,
                      ),
                    ),
                    Expanded(
                      child: Center(
                        child: AspectRatio(
                          aspectRatio: EngineConfig.analysisWidth /
                              EngineConfig.analysisHeight,
                          child: LayoutBuilder(
                            builder: (context, constraints) {
                              final size = Size(
                                  constraints.maxWidth, constraints.maxHeight);
                              return GestureDetector(
                                onTapUp: (d) => _onTapUp(d, size),
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    Image.memory(_frame!, fit: BoxFit.fill),
                                    CustomPaint(
                                      painter: _CornerPainter(_corners, size),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: done ? _analyze : null,
                          icon: const Icon(Icons.auto_awesome),
                          label: const Text('Analyze on device'),
                        ),
                      ),
                    ),
                  ],
                ),
    );
  }
}

class _CornerPainter extends CustomPainter {
  final List<List<double>> corners; // 960×540 space
  final Size renderSize;
  _CornerPainter(this.corners, this.renderSize);

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / EngineConfig.analysisWidth;
    final sy = size.height / EngineConfig.analysisHeight;
    final pts = [
      for (final c in corners) Offset(c[0] * sx, c[1] * sy)
    ];
    final dot = Paint()..color = Colors.tealAccent;
    final line = Paint()
      ..color = Colors.tealAccent
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    if (pts.length >= 2) {
      final path = Path()..addPolygon(pts, pts.length == 4);
      canvas.drawPath(path, line);
    }
    for (final p in pts) {
      canvas.drawCircle(p, 6, dot);
    }
  }

  @override
  bool shouldRepaint(_CornerPainter old) => old.corners.length != corners.length;
}
