import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';

import '../data/tennis_quotes.dart';
import '../theme.dart';

/// Progress bar paired with a slowly-rotating tennis quote, shown while a
/// match is uploading or being analyzed.
class QuoteProgress extends StatefulWidget {
  final String label;
  final double? value;

  const QuoteProgress({super.key, required this.label, this.value});

  @override
  State<QuoteProgress> createState() => _QuoteProgressState();
}

class _QuoteProgressState extends State<QuoteProgress> {
  late TennisQuote _quote;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _quote = tennisQuotes[Random().nextInt(tennisQuotes.length)];
    _timer = Timer.periodic(const Duration(seconds: 6), (_) => _nextQuote());
  }

  void _nextQuote() {
    if (!mounted) return;
    setState(() {
      _quote = tennisQuotes[Random().nextInt(tennisQuotes.length)];
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: widget.value,
            minHeight: 4,
          ),
        ),
        const SizedBox(height: 18),
        Text(
          widget.label,
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white, fontSize: 14),
        ),
        const SizedBox(height: 40),
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 500),
          child: Padding(
            key: ValueKey(_quote.text),
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              children: [
                Text(
                  '"${_quote.text}"',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontStyle: FontStyle.italic,
                    height: 1.4,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  '— ${_quote.author}',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: AppColors.skyBlue,
                    fontSize: 12,
                    letterSpacing: 1,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
