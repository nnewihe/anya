import 'package:flutter/widgets.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// The full Anya Tennis logo (ball + HUD frame + "ANYA" + tagline).
///
/// Uses the white / reversed one-color logo, which is drawn in white ink for
/// dark backgrounds — the app is dark-themed throughout. (Despite the file
/// name, `anya_logo_black.svg` is the *for-black-background* variant.)
class AnyaLogo extends StatelessWidget {
  final double width;

  const AnyaLogo({super.key, this.width = 240});

  @override
  Widget build(BuildContext context) {
    return SvgPicture.asset(
      'assets/images/anya_logo_black.svg',
      width: width,
    );
  }
}

/// The compact ball-and-HUD mark, without the wordmark — for tight spaces.
class AnyaBallMark extends StatelessWidget {
  final double size;

  const AnyaBallMark({super.key, this.size = 32});

  @override
  Widget build(BuildContext context) {
    return SvgPicture.asset(
      'assets/images/anya_ball_mark.svg',
      width: size,
      height: size,
    );
  }
}
