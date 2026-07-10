import 'package:gal/gal.dart';

class GalleryExportResult {
  final bool success;
  final String? error;
  const GalleryExportResult({required this.success, this.error});
}

/// Saves the generated rally reel to the native Photos/Gallery app.
class GalleryExportService {
  static Future<GalleryExportResult> saveVideo(String filePath) async {
    try {
      await Gal.requestAccess();
      await Gal.putVideo(filePath, album: 'Rally Predictor');
      return const GalleryExportResult(success: true);
    } on GalException catch (e) {
      return GalleryExportResult(success: false, error: _describe(e));
    } catch (e) {
      return GalleryExportResult(success: false, error: '$e');
    }
  }

  static String _describe(GalException e) {
    switch (e.type) {
      case GalExceptionType.accessDenied:
        return 'Permission denied — enable Photos access in Settings';
      case GalExceptionType.notEnoughSpace:
        return 'Not enough storage space';
      case GalExceptionType.notSupportedFormat:
        return 'Unsupported video format';
      default:
        return e.toString();
    }
  }
}
