import 'dart:convert';
import 'dart:io';

import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;

class YouTubeUploadResult {
  final bool success;
  final String? videoId;
  final String? error;
  const YouTubeUploadResult({required this.success, this.videoId, this.error});
}

/// YouTube video visibility. The wire values match the YouTube Data API v3
/// `status.privacyStatus` field exactly.
enum YouTubePrivacy {
  private,
  unlisted,
  public;

  String get wireValue => name; // 'private' | 'unlisted' | 'public'

  String get label => switch (this) {
        YouTubePrivacy.private => 'Private',
        YouTubePrivacy.unlisted => 'Unlisted',
        YouTubePrivacy.public => 'Public',
      };
}

/// Uploads the generated rally reel to YouTube via the resumable-upload flow
/// (POST to open a session, PUT the file bytes). The caller chooses the
/// visibility ([privacy]). Single-attempt: a failed PUT is not resumed, the
/// caller just retries from scratch — acceptable for the short clips this app
/// produces.
class YouTubeUploadService {
  static const _scopes = ['https://www.googleapis.com/auth/youtube.upload'];
  static const _initUrl =
      'https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status';

  static Future<YouTubeUploadResult> uploadVideo({
    required String filePath,
    required String title,
    YouTubePrivacy privacy = YouTubePrivacy.private,
    String description = '',
    void Function(double fraction)? onProgress,
  }) async {
    final GoogleSignInAccount? account;
    try {
      account = await GoogleSignIn(scopes: _scopes).signIn();
    } catch (e) {
      return YouTubeUploadResult(success: false, error: '$e');
    }
    if (account == null) {
      return const YouTubeUploadResult(success: false, error: 'cancelled');
    }

    try {
      final auth = await account.authentication;
      final token = auth.accessToken;
      if (token == null) {
        return const YouTubeUploadResult(
            success: false, error: 'No access token from sign-in');
      }

      final file = File(filePath);
      final length = await file.length();

      final initResp = await http.post(
        Uri.parse(_initUrl),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json; charset=UTF-8',
          'X-Upload-Content-Type': 'video/mp4',
          'X-Upload-Content-Length': '$length',
        },
        body: jsonEncode({
          'snippet': {'title': title, 'description': description},
          'status': {'privacyStatus': privacy.wireValue},
        }),
      );
      if (initResp.statusCode < 200 || initResp.statusCode >= 300) {
        return YouTubeUploadResult(
            success: false,
            error: 'Could not start upload (${initResp.statusCode})');
      }
      final uploadUrl = initResp.headers['location'];
      if (uploadUrl == null) {
        return const YouTubeUploadResult(
            success: false, error: 'No upload session returned');
      }

      final request = http.StreamedRequest('PUT', Uri.parse(uploadUrl))
        ..headers['Content-Type'] = 'video/mp4'
        ..headers['Content-Length'] = '$length';
      var sent = 0;
      file.openRead().listen(
        (chunk) {
          sent += chunk.length;
          onProgress?.call(length == 0 ? 1.0 : sent / length);
          request.sink.add(chunk);
        },
        onDone: () => request.sink.close(),
        onError: request.sink.addError,
        cancelOnError: true,
      );

      final streamedResp = await request.send();
      final body = await streamedResp.stream.bytesToString();
      if (streamedResp.statusCode < 200 || streamedResp.statusCode >= 300) {
        return YouTubeUploadResult(
            success: false,
            error: 'Upload failed (${streamedResp.statusCode})');
      }
      final videoId = (jsonDecode(body) as Map<String, dynamic>)['id'] as String?;
      return YouTubeUploadResult(success: true, videoId: videoId);
    } catch (e) {
      return YouTubeUploadResult(success: false, error: '$e');
    }
  }
}
