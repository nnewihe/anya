import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/api_client.dart';
import 'job_screen.dart';
import 'live_screen.dart';

/// Entry screen: choose a video source (gallery/file or live camera).
/// The system is source-agnostic — anything that produces an MP4 works.
///
/// Two upload modes:
///   • Single file — upload one MP4 (pre-concatenated match or single clip).
///   • Multi-clip  — pick multiple GoPro clips; server concatenates them in
///                   order before running the detector (mirrors run_pipeline.py).
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = ApiClient();
  bool _busy = false;
  String? _status;

  // ── Single-file upload ─────────────────────────────────────────────────
  Future<void> _pickAndUpload() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: false,
    );
    if (result == null || result.files.single.path == null) return;
    final file = File(result.files.single.path!);
    final filename = result.files.single.name;

    setState(() => _busy = true);
    try {
      final created = await _api.createJob(filename: filename, source: 'upload');
      if (!mounted) return;
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => JobScreen(
          api: _api,
          jobId: created.jobId,
          pendingUpload: PendingUpload(uploadUrl: created.uploadUrl, file: file),
        ),
      ));
    } on ApiException catch (e) {
      _snack('Failed to create job: ${e.statusCode}');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  // ── Multi-clip upload (GoPro clips → server concatenates) ──────────────
  Future<void> _pickClips() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: false,
      allowMultiple: true,
    );
    if (result == null || result.files.isEmpty) return;

    // Sort by filename so GH010897.MP4, GH020897.MP4 … come out in order.
    final files = result.files
        .where((f) => f.path != null)
        .toList()
      ..sort((a, b) => a.name.compareTo(b.name));

    if (files.isEmpty) return;

    setState(() {
      _busy = true;
      _status = 'Creating job…';
    });

    try {
      // Create the job in multi-clip mode (no single upload URL needed).
      final created = await _api.createJob(
        filename: 'match.mp4',
        source: 'gopro',
      );
      final jobId = created.jobId;

      // Register and upload each clip.
      for (int i = 0; i < files.length; i++) {
        final f = files[i];
        if (!mounted) return;
        setState(() => _status =
            'Uploading clip ${i + 1}/${files.length}: ${f.name}');

        final info = await _api.addClip(
          jobId: jobId,
          filename: f.name,
          clipIndex: i,
        );
        await _api.uploadVideo(
          info['upload_url']!,
          File(f.path!),
          onProgress: (p) {
            if (mounted) {
              setState(() => _status =
                  'Clip ${i + 1}/${files.length}: ${(p * 100).toStringAsFixed(0)}%');
            }
          },
        );
      }

      // All clips uploaded — start analysis.
      await _api.startJob(jobId);
      if (!mounted) return;
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => JobScreen(api: _api, jobId: jobId),
      ));
    } on ApiException catch (e) {
      _snack('Upload failed: ${e.statusCode} ${e.body}');
    } finally {
      if (mounted) setState(() { _busy = false; _status = null; });
    }
  }

  Future<void> _goLive() async {
    final created =
        await _api.createJob(filename: 'live.mp4', source: 'live');
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => LiveScreen(api: _api, jobId: created.jobId),
    ));
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Rally Predictor')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.sports_tennis, size: 96),
              const SizedBox(height: 12),
              Text('Detect rallies from any match',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 40),
              _BigButton(
                icon: Icons.video_library,
                label: 'Upload a match',
                subtitle: 'One file (pre-concatenated MP4)',
                onPressed: _busy ? null : _pickAndUpload,
              ),
              const SizedBox(height: 12),
              _BigButton(
                icon: Icons.video_collection,
                label: 'Upload GoPro clips',
                subtitle: 'Multiple clips — server stitches them',
                onPressed: _busy ? null : _pickClips,
              ),
              const SizedBox(height: 12),
              _BigButton(
                icon: Icons.videocam,
                label: 'Go live',
                subtitle: 'Record now and analyze on finish',
                onPressed: _busy ? null : _goLive,
              ),
              if (_busy) ...[
                const SizedBox(height: 24),
                const LinearProgressIndicator(),
                if (_status != null) ...[
                  const SizedBox(height: 8),
                  Text(_status!, textAlign: TextAlign.center),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _BigButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? subtitle;
  final VoidCallback? onPressed;

  const _BigButton({
    required this.icon,
    required this.label,
    this.subtitle,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 14),
          alignment: Alignment.centerLeft,
        ),
        onPressed: onPressed,
        icon: Icon(icon),
        label: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: const TextStyle(fontSize: 17)),
            if (subtitle != null)
              Text(subtitle!,
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.normal)),
          ],
        ),
      ),
    );
  }
}
