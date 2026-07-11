import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../widgets/anya_logo.dart';
import 'match_setup_screen.dart';

/// Entry screen for the on-device rally predictor. Pick a match video from the
/// device; everything (detection, tracking, reel) then runs locally — no
/// upload, no server, and no manual court calibration: the engine
/// self-calibrates its zones from where the players stand.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Future<void> _pickAndAnalyze(BuildContext context) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: false,
    );
    final path = result?.files.single.path;
    if (path == null) return;
    final name = result!.files.single.name;
    if (!context.mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => MatchSetupScreen(videoPath: path, videoName: name),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Anya Tennis')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const AnyaLogo(width: 240),
              const SizedBox(height: 28),
              Text(
                'Watch your matches in minutes, not hours.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 40),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  onPressed: () => _pickAndAnalyze(context),
                  icon: const Icon(Icons.video_library),
                  label: const Text('Choose a match video',
                      style: TextStyle(fontSize: 17)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
