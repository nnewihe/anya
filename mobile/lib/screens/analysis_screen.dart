import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../engine/engine.dart';

/// Runs the on-device engine on a picked video and shows progress, the detected
/// rally segments, and the resulting reel. Replaces the old upload → job → poll
/// flow entirely — no server, no upload.
class AnalysisScreen extends StatefulWidget {
  final String videoPath;
  final List<List<double>> corners;
  final String title;

  const AnalysisScreen({
    super.key,
    required this.videoPath,
    required this.corners,
    required this.title,
  });

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  double _progress = 0;
  String _message = 'Loading engine…';
  bool _done = false;
  String? _error;
  EngineResult? _result;
  VideoPlayerController? _video;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      final engine = await Engine.shared();
      final result = await engine.analyze(
        videoPath: widget.videoPath,
        corners: widget.corners,
        onProgress: (frac, msg) {
          if (mounted) {
            setState(() {
              _progress = frac;
              _message = msg;
            });
          }
        },
      );
      if (!mounted) return;
      setState(() {
        _result = result;
        _done = true;
      });
      if (result.reelPath != null) {
        await _initVideo(result.reelPath!);
      }
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  Future<void> _initVideo(String path) async {
    final c = VideoPlayerController.file(File(path));
    await c.initialize();
    if (!mounted) {
      await c.dispose();
      return;
    }
    setState(() => _video = c);
  }

  @override
  void dispose() {
    _video?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: Padding(padding: const EdgeInsets.all(20), child: _body()),
    );
  }

  Widget _body() {
    if (_error != null) {
      return _centered(Icons.error_outline, 'Analysis failed:\n$_error');
    }
    if (!_done) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            LinearProgressIndicator(value: _progress > 0 ? _progress : null),
            const SizedBox(height: 16),
            Text(_message, textAlign: TextAlign.center),
            const SizedBox(height: 8),
            Text('${(_progress * 100).toStringAsFixed(0)}%',
                style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      );
    }
    final segments = _result!.segments;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('${segments.length} ${segments.length == 1 ? "rally" : "rallies"} detected',
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 12),
        if (_video != null && _video!.value.isInitialized)
          _VideoBox(controller: _video!)
        else if (_result!.reelPath != null)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 24),
            child: Center(child: CircularProgressIndicator()),
          )
        else
          const Text('No reel (no segments met the detection criteria).'),
        const SizedBox(height: 12),
        Expanded(
          child: ListView.separated(
            itemCount: segments.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, i) {
              final s = segments[i];
              return ListTile(
                dense: true,
                leading: CircleAvatar(child: Text('${i + 1}')),
                title: Text('${_fmt(s.start)} → ${_fmt(s.end)}  '
                    '(${(s.end - s.start).toStringAsFixed(1)}s)'),
                trailing: Chip(label: Text(s.origin)),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _centered(IconData icon, String text) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 48),
            const SizedBox(height: 12),
            Text(text, textAlign: TextAlign.center),
          ],
        ),
      );

  static String _fmt(double sec) {
    final m = (sec ~/ 60).toString().padLeft(2, '0');
    final s = (sec % 60).toStringAsFixed(0).padLeft(2, '0');
    return '$m:$s';
  }
}

class _VideoBox extends StatefulWidget {
  final VideoPlayerController controller;
  const _VideoBox({required this.controller});

  @override
  State<_VideoBox> createState() => _VideoBoxState();
}

class _VideoBoxState extends State<_VideoBox> {
  void _toggle() {
    setState(() {
      widget.controller.value.isPlaying
          ? widget.controller.pause()
          : widget.controller.play();
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    return Column(
      children: [
        AspectRatio(
          aspectRatio: c.value.aspectRatio == 0 ? 16 / 9 : c.value.aspectRatio,
          child: VideoPlayer(c),
        ),
        IconButton(
          iconSize: 40,
          icon: Icon(c.value.isPlaying ? Icons.pause_circle : Icons.play_circle),
          onPressed: _toggle,
        ),
      ],
    );
  }
}
