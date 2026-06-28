import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';
import '../models/job.dart';

/// Talks to the rally-predictor backend.
///
/// Upload flow:
///   1. createJob() → {jobId, uploadUrl}
///   2. uploadVideo(uploadUrl, file) → PUT raw bytes (S3 presigned or local)
///   3. startJob(jobId) → enqueue analysis
///   4. watchJob(jobId) → stream of Job states over WebSocket
class ApiClient {
  final String baseUrl;
  final Dio _dio;

  ApiClient({String? baseUrl})
      : baseUrl = baseUrl ?? AppConfig.apiBaseUrl,
        _dio = Dio();

  Uri _u(String path) => Uri.parse('$baseUrl$path');

  /// Resolve an upload/result URL that may be absolute (S3) or relative (dev).
  String _resolve(String url) =>
      url.startsWith('http') ? url : '$baseUrl$url';

  // ── Job lifecycle ──────────────────────────────────────────────────────
  Future<CreateJobResponse> createJob({
    required String filename,
    String contentType = 'video/mp4',
    String source = 'upload',
  }) async {
    final res = await http.post(
      _u('/jobs'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'filename': filename,
        'content_type': contentType,
        'source': source,
      }),
    );
    _ensureOk(res);
    return CreateJobResponse.fromJson(jsonDecode(res.body));
  }

  /// PUT the file to the (presigned or local) upload URL, reporting 0..1 progress.
  Future<void> uploadVideo(
    String uploadUrl,
    File file, {
    void Function(double progress)? onProgress,
    String contentType = 'video/mp4',
  }) async {
    final len = await file.length();
    await _dio.putUri(
      Uri.parse(_resolve(uploadUrl)),
      data: file.openRead(),
      options: Options(
        headers: {
          Headers.contentLengthHeader: len,
          'Content-Type': contentType,
        },
      ),
      onSendProgress: (sent, total) {
        if (onProgress != null && total > 0) onProgress(sent / total);
      },
    );
  }

  Future<Job> startJob(String jobId) async {
    final res = await http.post(_u('/jobs/$jobId/start'));
    _ensureOk(res);
    return Job.fromJson(jsonDecode(res.body));
  }

  Future<Job> getJob(String jobId) async {
    final res = await http.get(_u('/jobs/$jobId'));
    _ensureOk(res);
    return Job.fromJson(jsonDecode(res.body));
  }

  /// Live job-state stream over WebSocket. Falls back to nothing on error —
  /// callers should also poll getJob() as a safety net.
  Stream<Job> watchJob(String jobId) {
    final ws = WebSocketChannel.connect(
      Uri.parse('${AppConfig.wsBaseUrl}/jobs/$jobId/events'),
    );
    return ws.stream
        .map((msg) => jsonDecode(msg as String) as Map<String, dynamic>)
        .where((m) => m.containsKey('job_id')) // skip keepalive frames
        .map(Job.fromJson);
  }

  /// Absolute URL for the finished rally reel (for video_player / download).
  String resolveResultUrl(String resultUrl) => _resolve(resultUrl);

  // ── Multi-clip upload ──────────────────────────────────────────────────
  /// Register one clip, returning its presigned upload URL.
  Future<Map<String, String>> addClip({
    required String jobId,
    required String filename,
    required int clipIndex,
    String contentType = 'video/mp4',
  }) async {
    final res = await http.post(
      _u('/jobs/$jobId/clips'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'filename': filename,
        'content_type': contentType,
        'clip_index': clipIndex,
      }),
    );
    _ensureOk(res);
    final j = jsonDecode(res.body) as Map<String, dynamic>;
    return {
      'clip_key': j['clip_key'] as String,
      'upload_url': j['upload_url'] as String,
    };
  }

  // ── Live streaming ─────────────────────────────────────────────────────
  /// Open the live ingest socket. The caller writes binary video chunks and
  /// sends the text "EOS" (or closes) when the match ends.
  WebSocketChannel openLiveSocket(String jobId) {
    return WebSocketChannel.connect(
      Uri.parse('${AppConfig.wsBaseUrl}/live/$jobId'),
    );
  }

  void _ensureOk(http.Response res) {
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw ApiException(res.statusCode, res.body);
    }
  }
}

class ApiException implements Exception {
  final int statusCode;
  final String body;
  ApiException(this.statusCode, this.body);
  @override
  String toString() => 'ApiException($statusCode): $body';
}
