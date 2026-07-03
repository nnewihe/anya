import 'package:flutter/widgets.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// The full Anya Tennis wordmark (ball + HUD frame + "ANYA" + tagline).
class AnyaLogo extends StatelessWidget {
  final double width;

  const AnyaLogo({super.key, this.width = 240});

  @override
  Widget build(BuildContext context) {
    return SvgPicture.asset(
      'assets/images/anya_logo_mark.svg',
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
