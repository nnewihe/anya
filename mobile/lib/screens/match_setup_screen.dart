import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../engine/engine.dart';
import '../engine/engine_isolate.dart';
import '../services/background_analysis.dart';
import '../services/gallery_export.dart';
import '../services/youtube_config.dart';
import '../services/youtube_upload.dart';

/// Shown immediately after a match video is chosen. Two things happen at once:
///
///  1. The on-device engine starts analysing the video right away — no "start"
///     button. A slim progress bar reports how far along it is.
///  2. The user fills in the match details — title, the two players' names, and
///     the YouTube upload privacy — while the analysis runs.
///
/// When analysis finishes, the reel + segment list + share actions appear in
/// place, using the title / privacy the user set. No business logic changes:
/// the engine call is identical to before, just kicked off up front.
class MatchSetupScreen extends StatefulWidget {
  /// Absolute path to the picked match video.
  final String videoPath;

  /// The picked file's name (unused for the title now, kept for reference).
  final String videoName;

  const MatchSetupScreen({
    super.key,
    required this.videoPath,
    required this.videoName,
  });

  @override
  State<MatchSetupScreen> createState() => _MatchSetupScreenState();
}

class _MatchSetupScreenState extends State<MatchSetupScreen>
    with WidgetsBindingObserver {
  // --- Engine / analysis state ---
  double _progress = 0;
  String _message = 'Loading engine…';
  bool _done = false;
  String? _error;
  EngineResult? _result;
  VideoPlayerController? _video;

  /// Whether the app is currently foregrounded (drives the completion
  /// notification: only notify if the user isn't looking at the screen).
  bool _appInForeground = true;

  /// Last whole-percent pushed to the background notification, to avoid
  /// spamming the service with an update on every frame.
  int _lastNotifiedPct = -1;

  // --- Match-detail form state ---
  final _titleController = TextEditingController();
  final _playerAController = TextEditingController();
  final _playerBController = TextEditingController();
  YouTubePrivacy _privacy = YouTubePrivacy.private;

  /// True once the user has hand-edited the title, after which we stop
  /// regenerating it from the player names.
  bool _titleEdited = false;

  /// Guards our own programmatic writes to the title field so they aren't
  /// mistaken for a manual edit.
  bool _settingTitleProgrammatically = false;

  // --- Share state ---
  bool _savingToGallery = false;
  bool _uploadingToYouTube = false;
  double _uploadProgress = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _syncDefaultTitle();
    _titleController.addListener(_onTitleChanged);
    _playerAController.addListener(_syncDefaultTitle);
    _playerBController.addListener(_syncDefaultTitle);
    _run();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Leaving the screen ends the job's UI; clear any lingering service.
    BackgroundAnalysis.stop();
    _titleController.dispose();
    _playerAController.dispose();
    _playerBController.dispose();
    _video?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _appInForeground = state == AppLifecycleState.resumed;
    // When the user comes back and analysis has already finished, clear the
    // "reel is ready" notification.
    if (_appInForeground && _done) {
      BackgroundAnalysis.stop();
    }
  }

  // ---------------------------------------------------------------------------
  // Title defaulting
  // ---------------------------------------------------------------------------

  static String _todayIso() {
    final now = DateTime.now();
    final m = now.month.toString().padLeft(2, '0');
    final d = now.day.toString().padLeft(2, '0');
    return '${now.year}-$m-$d';
  }

  String _playerAName() =>
      _playerAController.text.trim().isEmpty ? 'Player A' : _playerAController.text.trim();

  String _playerBName() =>
      _playerBController.text.trim().isEmpty ? 'Player B' : _playerBController.text.trim();

  String _defaultTitle() =>
      '${_todayIso()} Tennis Match — ${_playerAName()} vs ${_playerBName()}';

  /// Rewrite the title from the current date + player names, unless the user
  /// has taken the title over with a manual edit.
  void _syncDefaultTitle() {
    if (_titleEdited) return;
    final next = _defaultTitle();
    if (_titleController.text == next) return;
    _settingTitleProgrammatically = true;
    _titleController.value = TextEditingValue(
      text: next,
      selection: TextSelection.collapsed(offset: next.length),
    );
    _settingTitleProgrammatically = false;
  }

  void _onTitleChanged() {
    if (_settingTitleProgrammatically) return;
    // A change that didn't come from _syncDefaultTitle is a manual edit.
    if (!_titleEdited) setState(() => _titleEdited = true);
  }

  // ---------------------------------------------------------------------------
  // Engine
  // ---------------------------------------------------------------------------

  Future<void> _run() async {
    // Keep analysis alive when the app is backgrounded (best-effort within OS
    // limits) and surface progress in a notification. No-op on desktop/web.
    await BackgroundAnalysis.requestPermissions();
    await BackgroundAnalysis.start();
    try {
      final result = await analyzeInBackground(
        videoPath: widget.videoPath,
        onProgress: (frac, msg) {
          if (mounted) {
            setState(() {
              _progress = frac;
              _message = msg;
            });
          }
          _pushNotificationProgress(frac, msg);
        },
      );
      if (!mounted) return;
      setState(() {
        _result = result;
        _done = true;
      });
      await _onAnalysisComplete(result);
      if (result.reelPath != null) {
        await _initVideo(result.reelPath!);
      }
    } catch (e) {
      await BackgroundAnalysis.stop();
      if (mounted) setState(() => _error = '$e');
    }
  }

  void _pushNotificationProgress(double frac, String msg) {
    final pct = (frac.clamp(0.0, 1.0) * 100).floor();
    if (pct == _lastNotifiedPct) return;
    _lastNotifiedPct = pct;
    // Fire-and-forget; the service tolerates rapid updates and we've throttled
    // to whole-percent steps.
    BackgroundAnalysis.update(frac, msg);
  }

  Future<void> _onAnalysisComplete(EngineResult result) async {
    final count = result.segments.length;
    final noun = count == 1 ? 'rally' : 'rallies';
    if (_appInForeground) {
      // User is looking at the screen — no need for a notification.
      await BackgroundAnalysis.stop();
    } else {
      await BackgroundAnalysis.complete(
        message: '$count $noun detected. Tap to view your reel.',
      );
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

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Anya Tennis')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            _statusBar(),
            const SizedBox(height: 24),
            _form(),
            if (_error != null) ...[
              const SizedBox(height: 24),
              _errorBox(_error!),
            ] else if (_done) ...[
              const SizedBox(height: 24),
              const Divider(),
              const SizedBox(height: 12),
              _result != null ? _resultSection(_result!) : const SizedBox.shrink(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _statusBar() {
    if (_error != null) {
      return const SizedBox.shrink();
    }
    if (_done) {
      return Row(
        children: [
          const Icon(Icons.check_circle, size: 18),
          const SizedBox(width: 8),
          Text('Analysis complete',
              style: Theme.of(context).textTheme.bodyMedium),
        ],
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(_message,
                style: Theme.of(context).textTheme.bodyMedium,
                overflow: TextOverflow.ellipsis),
            const SizedBox(width: 12),
            Text('${(_progress * 100).toStringAsFixed(0)}%',
                style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
        const SizedBox(height: 8),
        LinearProgressIndicator(value: _progress > 0 ? _progress : null),
      ],
    );
  }

  Widget _form() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _label('Title'),
        const SizedBox(height: 6),
        TextField(
          controller: _titleController,
          textInputAction: TextInputAction.next,
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            isDense: true,
          ),
        ),
        const SizedBox(height: 20),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _label('Player A'),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _playerAController,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      hintText: 'Player A',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _label('Player B'),
                  const SizedBox(height: 6),
                  TextField(
                    controller: _playerBController,
                    textInputAction: TextInputAction.done,
                    decoration: const InputDecoration(
                      hintText: 'Player B',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),
        _label('Upload privacy'),
        const SizedBox(height: 6),
        _privacySelector(),
      ],
    );
  }

  Widget _label(String text) => Text(
        text.toUpperCase(),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(
              letterSpacing: 1.5,
              color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.7),
            ),
      );

  Widget _privacySelector() {
    return SegmentedButton<YouTubePrivacy>(
      segments: const [
        ButtonSegment(
          value: YouTubePrivacy.private,
          label: Text('Private'),
          icon: Icon(Icons.lock_outline, size: 18),
        ),
        ButtonSegment(
          value: YouTubePrivacy.unlisted,
          label: Text('Unlisted'),
          icon: Icon(Icons.link, size: 18),
        ),
        ButtonSegment(
          value: YouTubePrivacy.public,
          label: Text('Public'),
          icon: Icon(Icons.public, size: 18),
        ),
      ],
      selected: {_privacy},
      showSelectedIcon: false,
      onSelectionChanged: (s) => setState(() => _privacy = s.first),
    );
  }

  Widget _errorBox(String error) => Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline, size: 24),
          const SizedBox(width: 12),
          Expanded(child: Text('Analysis failed:\n$error')),
        ],
      );

  Widget _resultSection(EngineResult result) {
    final segments = result.segments;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '${segments.length} ${segments.length == 1 ? "rally" : "rallies"} detected',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 12),
        if (_video != null && _video!.value.isInitialized)
          _VideoBox(controller: _video!)
        else if (result.reelPath != null)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 24),
            child: Center(child: CircularProgressIndicator()),
          )
        else
          const Text('No reel (no segments met the detection criteria).'),
        if (result.reelPath != null) ...[
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: _savingToGallery
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.save_alt),
                  label: Text(_savingToGallery ? 'Saving…' : 'Save to Gallery'),
                  onPressed: _savingToGallery ? null : _onSaveToGallery,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  icon: _uploadingToYouTube
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.upload),
                  label: Text(_uploadingToYouTube
                      ? 'Uploading ${(_uploadProgress * 100).toStringAsFixed(0)}%'
                      : 'Upload to YouTube'),
                  onPressed: _uploadingToYouTube ? null : _onUploadToYouTube,
                ),
              ),
            ],
          ),
        ],
        const SizedBox(height: 12),
        for (var i = 0; i < segments.length; i++) ...[
          if (i > 0) const Divider(height: 1),
          _segmentTile(i, segments[i]),
        ],
      ],
    );
  }

  Widget _segmentTile(int i, RallySegment s) => ListTile(
        dense: true,
        contentPadding: EdgeInsets.zero,
        leading: CircleAvatar(child: Text('${i + 1}')),
        title: Text('${_fmt(s.start)} → ${_fmt(s.end)}  '
            '(${(s.end - s.start).toStringAsFixed(1)}s)'),
        trailing: Chip(label: Text(s.origin)),
      );

  // ---------------------------------------------------------------------------
  // Share actions
  // ---------------------------------------------------------------------------

  Future<void> _onSaveToGallery() async {
    setState(() => _savingToGallery = true);
    final result = await GalleryExportService.saveVideo(_result!.reelPath!);
    if (!mounted) return;
    setState(() => _savingToGallery = false);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(
          result.success ? 'Saved to gallery' : 'Save failed: ${result.error}'),
    ));
  }

  Future<void> _onUploadToYouTube() async {
    if (!kYouTubeUploadConfigured) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('YouTube upload needs a one-time Google OAuth setup — '
            'see docs/sharing_setup.md'),
        duration: Duration(seconds: 5),
      ));
      return;
    }
    setState(() {
      _uploadingToYouTube = true;
      _uploadProgress = 0;
    });
    final title = _titleController.text.trim().isEmpty
        ? _defaultTitle()
        : _titleController.text.trim();
    final result = await YouTubeUploadService.uploadVideo(
      filePath: _result!.reelPath!,
      title: title,
      privacy: _privacy,
      description: '${_playerAName()} vs ${_playerBName()} — rally reel, '
          'generated by Anya Tennis.',
      onProgress: (f) {
        if (mounted) setState(() => _uploadProgress = f);
      },
    );
    if (!mounted) return;
    setState(() => _uploadingToYouTube = false);
    final message = result.success
        ? 'Uploaded to YouTube (${_privacy.label.toLowerCase()})'
        : result.error == 'cancelled'
            ? 'Upload cancelled'
            : 'Upload failed: ${result.error}';
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

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
