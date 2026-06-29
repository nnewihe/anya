/// Dart mirror of backend/app/schemas.py. Keep field names in sync.

enum JobStatus { pending, queued, processing, completed, failed, unknown }

JobStatus _statusFromString(String? s) {
  switch (s) {
    case 'pending':
      return JobStatus.pending;
    case 'queued':
      return JobStatus.queued;
    case 'processing':
      return JobStatus.processing;
    case 'completed':
      return JobStatus.completed;
    case 'failed':
      return JobStatus.failed;
    default:
      return JobStatus.unknown;
  }
}

class Segment {
  final double start;
  final double end;
  final String origin;

  Segment({required this.start, required this.end, required this.origin});

  factory Segment.fromJson(Map<String, dynamic> j) => Segment(
        start: (j['start'] as num).toDouble(),
        end: (j['end'] as num).toDouble(),
        origin: j['origin'] as String? ?? 'near',
      );

  double get duration => end - start;
}

class Job {
  final String jobId;
  final JobStatus status;
  final String source;
  final String? filename;
  final List<String> clipKeys;
  final double progress; // 0..1
  final String? message;
  final List<Segment> segments;
  final String? resultUrl;
  final String? error;

  Job({
    required this.jobId,
    required this.status,
    required this.source,
    required this.filename,
    required this.clipKeys,
    required this.progress,
    required this.message,
    required this.segments,
    required this.resultUrl,
    required this.error,
  });

  factory Job.fromJson(Map<String, dynamic> j) => Job(
        jobId: j['job_id'] as String,
        status: _statusFromString(j['status'] as String?),
        source: j['source'] as String? ?? 'upload',
        filename: j['filename'] as String?,
        clipKeys: ((j['clip_keys'] as List?) ?? []).cast<String>(),
        progress: (j['progress'] as num?)?.toDouble() ?? 0.0,
        message: j['message'] as String?,
        segments: ((j['segments'] as List?) ?? [])
            .map((e) => Segment.fromJson(e as Map<String, dynamic>))
            .toList(),
        resultUrl: j['result_url'] as String?,
        error: j['error'] as String?,
      );

  bool get isTerminal =>
      status == JobStatus.completed || status == JobStatus.failed;
}

class CreateJobResponse {
  final String jobId;
  final String uploadUrl;
  final String uploadMethod;

  CreateJobResponse({
    required this.jobId,
    required this.uploadUrl,
    required this.uploadMethod,
  });

  factory CreateJobResponse.fromJson(Map<String, dynamic> j) =>
      CreateJobResponse(
        jobId: j['job_id'] as String,
        uploadUrl: j['upload_url'] as String,
        uploadMethod: j['upload_method'] as String? ?? 'PUT',
      );
}
