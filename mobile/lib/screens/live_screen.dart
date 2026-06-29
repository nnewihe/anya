import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../api/api_client.dart';
import 'job_screen.dart';

/// Live capture screen.
///
/// v1 strategy ("near-real-time"): record the match with the device camera,
/// then stream the recording to the backend's /live/{job} WebSocket in chunks
/// when you stop. The backend assembles the chunks and runs the same
/// rally_detector pipeline, so results appear seconds after you finish.
///
/// True frame-by-frame live detection (WebRTC/HLS + a rolling-buffer detector)
/// is documented as future work in backend/README.md.
class LiveScreen extends StatefulWidget {
  final ApiClient api;
  final String jobId;

  const LiveScreen({super.key, required this.api, required this.jobId});

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> {
  CameraController? _controller;
  bool _recording = false;
  bool _streaming = false;
  double _streamProgress = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera available');
        return;
      }
      final controller = CameraController(
        cameras.first,
        ResolutionPreset.high,
        enableAudio: true,
      );
      await controller.initialize();
      if (!mounted) return;
      setState(() => _controller = controller);
    } catch (e) {
      setState(() => _error = '$e');
    }
  }

  Future<void> _toggleRecording() async {
    final c = _controller;
    if (c == null) return;

    if (!_recording) {
      await c.startVideoRecording();
      setState(() => _recording = true);
    } else {
      final file = await c.stopVideoRecording();
      setState(() => _recording = false);
      await _streamToBackend(File(file.path));
    }
  }

  /// Stream the recorded file to /live/{job} in chunks, then open the job view.
  Future<void> _streamToBackend(File file) async {
    setState(() => _streaming = true);
    final socket = widget.api.openLiveSocket(widget.jobId);

    try {
      final total = await file.length();
      var sent = 0;
      // 256 KiB chunks keep memory flat for multi-GB recordings.
      await for (final chunk in file.openRead().transform(_chunker(256 * 1024))) {
        socket.sink.add(chunk);
        sent += chunk.length;
        if (mounted) setState(() => _streamProgress = sent / total);
      }
      socket.sink.add('EOS');
      await socket.sink.close();

      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => JobScreen(api: widget.api, jobId: widget.jobId),
      ));
    } catch (e) {
      setState(() {
        _streaming = false;
        _error = '$e';
      });
    }
  }

  /// Re-chunk an arbitrary byte stream into fixed-size pieces.
  StreamTransformer<List<int>, List<int>> _chunker(int size) {
    final buffer = <int>[];
    return StreamTransformer.fromHandlers(
      handleData: (data, sink) {
        buffer.addAll(data);
        while (buffer.length >= size) {
          sink.add(buffer.sublist(0, size));
          buffer.removeRange(0, size);
        }
      },
      handleDone: (sink) {
        if (buffer.isNotEmpty) sink.add(List<int>.from(buffer));
        sink.close();
      },
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Live capture')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error != null) {
      return Center(child: Text('Error: $_error'));
    }
    if (_streaming) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(value: _streamProgress),
            const SizedBox(height: 16),
            Text('Uploading match… '
                '${(_streamProgress * 100).toStringAsFixed(0)}%'),
          ],
        ),
      );
    }
    final c = _controller;
    if (c == null || !c.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return Stack(
      alignment: Alignment.bottomCenter,
      children: [
        Positioned.fill(child: CameraPreview(c)),
        Padding(
          padding: const EdgeInsets.only(bottom: 40),
          child: FloatingActionButton.large(
            backgroundColor: _recording ? Colors.red : Colors.white,
            onPressed: _toggleRecording,
            child: Icon(
              _recording ? Icons.stop : Icons.fiber_manual_record,
              color: _recording ? Colors.white : Colors.red,
              size: 40,
            ),
          ),
        ),
        if (_recording)
          const Positioned(
            top: 20,
            child: Chip(
              backgroundColor: Colors.red,
              label: Text('REC', style: TextStyle(color: Colors.white)),
            ),
          ),
      ],
    );
  }
}
