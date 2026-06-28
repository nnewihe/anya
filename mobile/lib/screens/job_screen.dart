import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../api/api_client.dart';
import '../models/job.dart';

/// Work to perform when the JobScreen opens for an upload job: PUT the file,
/// then start analysis.
class PendingUpload {
  final String uploadUrl;
  final File file;
  PendingUpload({required this.uploadUrl, required this.file});
}

/// Shows the lifecycle of a single job: upload → queued → processing →
/// completed (with the rally reel) or failed.
class JobScreen extends StatefulWidget {
  final ApiClient api;
  final String jobId;
  final PendingUpload? pendingUpload;

  const JobScreen({
    super.key,
    required this.api,
    required this.jobId,
    this.pendingUpload,
  });

  @override
  State<JobScreen> createState() => _JobScreenState();
}

class _JobScreenState extends State<JobScreen> {
  Job? _job;
  double _uploadProgress = 0;
  bool _uploading = false;
  String? _fatal;

  StreamSubscription<Job>? _sub;
  Timer? _pollTimer;
  VideoPlayerController? _video;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      if (widget.pendingUpload != null) {
        setState(() => _uploading = true);
        await widget.api.uploadVideo(
          widget.pendingUpload!.uploadUrl,
          widget.pendingUpload!.file,
          onProgress: (p) => setState(() => _uploadProgress = p),
        );
        await widget.api.startJob(widget.jobId);
        setState(() => _uploading = false);
      }
      _watch();
    } catch (e) {
      setState(() => _fatal = '$e');
    }
  }

  void _watch() {
    // Primary: WebSocket push.
    _sub = widget.api.watchJob(widget.jobId).listen(
          _onJob,
          onError: (_) {}, // polling below covers WS failures
        );
    // Safety net: poll every 5s in case the socket drops.
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (_) async {
      if (_job?.isTerminal == true) return;
      try {
        _onJob(await widget.api.getJob(widget.jobId));
      } catch (_) {}
    });
  }

  void _onJob(Job job) {
    setState(() => _job = job);
    if (job.status == JobStatus.completed && job.resultUrl != null) {
      _initVideo(widget.api.resolveResultUrl(job.resultUrl!));
      _stopWatching();
    } else if (job.status == JobStatus.failed) {
      _stopWatching();
    }
  }

  void _stopWatching() {
    _sub?.cancel();
    _pollTimer?.cancel();
  }

  Future<void> _initVideo(String url) async {
    final c = VideoPlayerController.networkUrl(Uri.parse(url));
    await c.initialize();
    if (!mounted) {
      await c.dispose();
      return;
    }
    setState(() => _video = c);
  }

  @override
  void dispose() {
    _stopWatching();
    _video?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Match ${widget.jobId.substring(0, 8)}')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: _buildBody(),
      ),
    );
  }

  Widget _buildBody() {
    if (_fatal != null) {
      return _Centered(icon: Icons.error_outline, text: _fatal!);
    }
    if (_uploading) {
      return _Progress(
        label: 'Uploading… ${(_uploadProgress * 100).toStringAsFixed(0)}%',
        value: _uploadProgress,
      );
    }
    final job = _job;
    if (job == null) {
      return const _Progress(label: 'Connecting…', value: null);
    }

    switch (job.status) {
      case JobStatus.completed:
        return _Completed(job: job, video: _video);
      case JobStatus.failed:
        return _Centered(
          icon: Icons.error_outline,
          text: 'Analysis failed:\n${job.error ?? job.message ?? "unknown"}',
        );
      case JobStatus.processing:
        return _Progress(
          label: job.message ?? 'Analyzing…',
          value: job.progress > 0 ? job.progress : null,
        );
      default:
        return _Progress(label: job.message ?? 'Queued…', value: null);
    }
  }
}

class _Progress extends StatelessWidget {
  final String label;
  final double? value;
  const _Progress({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          LinearProgressIndicator(value: value),
          const SizedBox(height: 16),
          Text(label, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

class _Completed extends StatelessWidget {
  final Job job;
  final VideoPlayerController? video;
  const _Completed({required this.job, required this.video});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('${job.segments.length} rallies detected',
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 12),
        if (video != null && video!.value.isInitialized)
          _VideoBox(controller: video!)
        else
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 24),
            child: Center(child: CircularProgressIndicator()),
          ),
        const SizedBox(height: 12),
        Expanded(
          child: ListView.separated(
            itemCount: job.segments.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, i) {
              final s = job.segments[i];
              return ListTile(
                dense: true,
                leading: CircleAvatar(child: Text('${i + 1}')),
                title: Text(
                    '${_fmt(s.start)} → ${_fmt(s.end)}  (${s.duration.toStringAsFixed(1)}s)'),
                trailing: Chip(label: Text(s.origin)),
              );
            },
          ),
        ),
      ],
    );
  }

  static String _fmt(double sec) {
    final m = (sec ~/ 60).toString().padLeft(2, '0');
    final s = (sec % 60).toStringAsFixed(0).padLeft(2, '0');
    return '$m:$s';
  }
}

class _VideoBox extends StatelessWidget {
  final VideoPlayerController controller;
  const _VideoBox({required this.controller});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        AspectRatio(
          aspectRatio: controller.value.aspectRatio == 0
              ? 16 / 9
              : controller.value.aspectRatio,
          child: VideoPlayer(controller),
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            IconButton(
              iconSize: 40,
              icon: Icon(controller.value.isPlaying
                  ? Icons.pause_circle
                  : Icons.play_circle),
              onPressed: () {
                controller.value.isPlaying
                    ? controller.pause()
                    : controller.play();
              },
            ),
          ],
        ),
      ],
    );
  }
}

class _Centered extends StatelessWidget {
  final IconData icon;
  final String text;
  const _Centered({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 48),
          const SizedBox(height: 12),
          Text(text, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}
