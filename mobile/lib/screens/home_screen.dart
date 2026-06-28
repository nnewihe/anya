import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../models/job.dart';
import 'job_screen.dart';
import 'live_screen.dart';

/// Entry screen: choose a video source (gallery/file or live camera).
/// The system is source-agnostic — anything that produces an MP4 works.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = ApiClient();
  bool _busy = false;

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
      // 1. Create the job and get an upload target.
      final created =
          await _api.createJob(filename: filename, source: 'upload');

      if (!mounted) return;
      // 2. Hand off to the job screen, which uploads + watches progress.
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => JobScreen(
          api: _api,
          jobId: created.jobId,
          // Pass the upload work so the JobScreen can show upload progress.
          pendingUpload: PendingUpload(
            uploadUrl: created.uploadUrl,
            file: file,
          ),
        ),
      ));
    } on ApiException catch (e) {
      _snack('Failed to create job: ${e.statusCode}');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _goLive() async {
    final created = await _api.createJob(
      filename: 'live.mp4',
      source: 'live',
    );
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
                onPressed: _busy ? null : _pickAndUpload,
              ),
              const SizedBox(height: 16),
              _BigButton(
                icon: Icons.videocam,
                label: 'Go live',
                onPressed: _busy ? null : _goLive,
              ),
              if (_busy) ...[
                const SizedBox(height: 32),
                const CircularProgressIndicator(),
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
  final VoidCallback? onPressed;

  const _BigButton({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 20),
        ),
        onPressed: onPressed,
        icon: Icon(icon),
        label: Text(label, style: const TextStyle(fontSize: 18)),
      ),
    );
  }
}
