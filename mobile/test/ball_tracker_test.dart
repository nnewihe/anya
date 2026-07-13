import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:anya_tennis/engine/ball_tracker.dart';

// Port of pipeline/ball_tracker.py's _run_self_test — the parity oracle for the
// IMM Kalman ball tracker. Same 10 scenarios and the same expected outcomes.

const double fps = 30.0;
const double dt = 1.0 / fps;

List<TrackStatus> run(List<List<Detection>> stream, {PerspectiveScale? persp}) {
  final mgr = BallTrackManager(fps, perspectiveScale: persp);
  final out = <TrackStatus>[];
  var t = 0.0;
  for (final dets in stream) {
    out.add(mgr.update(dets, t));
    t += dt;
  }
  return out;
}

List<Detection> d1(double x, double y, double c) => [Detection(x, y, c)];

void main() {
  test('S1: moving ball becomes and stays a live trace', () {
    final moving = <List<Detection>>[];
    var x = 100.0;
    const y = 300.0;
    for (var i = 0; i < 40; i++) {
      x += 18.0;
      moving.add(d1(x, y, 0.9));
    }
    final res = run(moving);
    expect(res.any((s) => s.hasMovingTrace), isTrue);
    expect(res.last.hasMovingTrace, isTrue);
    expect(res.last.state, 'moving');
  });

  test('S2: ball that stops in view ends the trace', () {
    final stop = <List<Detection>>[];
    var x = 100.0;
    for (var i = 0; i < 25; i++) {
      x += 18.0;
      stop.add(d1(x, 300.0, 0.9));
    }
    for (var i = 0; i < 25; i++) {
      stop.add(d1(x, 300.0, 0.9));
    }
    final res = run(stop);
    expect(res.last.hasMovingTrace, isFalse);
    expect(res.last.state, 'stopped');
  });

  test('S3: disappearing ball ends within miss_timeout', () {
    final dis = <List<Detection>>[];
    var x = 100.0;
    for (var i = 0; i < 25; i++) {
      x += 18.0;
      dis.add(d1(x, 300.0, 0.9));
    }
    for (var i = 0; i < 100; i++) {
      dis.add(const []);
    }
    final res = run(dis);
    expect(res.last.hasMovingTrace, isFalse);
    final aliveIdx = [
      for (var i = 0; i < res.length; i++)
        if (res[i].hasMovingTrace) i
    ];
    final lastAlive = aliveIdx.isNotEmpty ? aliveIdx.last : 24;
    final coastS = (lastAlive - 24) * dt;
    expect(coastS <= 2.0 + dt + 1e-9, isTrue);
    expect(lastAlive >= 24, isTrue);
  });

  test('S4: scattered false positives never form a live trace', () {
    final rng = math.Random(0);
    final fp = <List<Detection>>[];
    for (var i = 0; i < 40; i++) {
      if (rng.nextDouble() < 0.4) {
        fp.add(d1(rng.nextInt(900).toDouble(), rng.nextInt(500).toDouble(), 0.3));
      } else {
        fp.add(const []);
      }
    }
    final res = run(fp);
    expect(res.any((s) => s.hasMovingTrace), isFalse);
  });

  test('S5: a stationary ball never becomes a live trace', () {
    final stat = [for (var i = 0; i < 40; i++) d1(500.0, 250.0, 0.8)];
    final res = run(stat);
    expect(res.any((s) => s.hasMovingTrace), isFalse);
  });

  test('S6: trace survives a brief occlusion', () {
    final occ = <List<Detection>>[];
    var x = 100.0;
    for (var i = 0; i < 20; i++) {
      x += 18.0;
      occ.add(d1(x, 300.0, 0.9));
    }
    for (var i = 0; i < 6; i++) {
      x += 18.0;
      occ.add(const []);
    }
    for (var i = 0; i < 20; i++) {
      x += 18.0;
      occ.add(d1(x, 300.0, 0.9));
    }
    final res = run(occ);
    expect(res.last.hasMovingTrace, isTrue);
    expect(res.sublist(25).every((s) => s.hasMovingTrace), isTrue);
  });

  test('S7: ball stays alive through a 180-degree reversal (racket)', () {
    final rev = <List<Detection>>[];
    var x = 200.0;
    for (var i = 0; i < 25; i++) {
      x += 30.0;
      rev.add(d1(x, 300.0, 0.9));
    }
    rev.add(const []);
    for (var i = 0; i < 25; i++) {
      x -= 30.0;
      rev.add(d1(x, 300.0, 0.9));
    }
    final res = run(rev);
    final aliveAfter = [for (final s in res.sublist(26)) s.hasMovingTrace];
    var maxDead = 0, cur = 0;
    for (final a in aliveAfter) {
      cur = a ? 0 : cur + 1;
      maxDead = math.max(maxDead, cur);
    }
    expect(aliveAfter.sublist(aliveAfter.length - 10).every((a) => a), isTrue);
    expect(maxDead <= 2, isTrue);
    final mp = [for (final s in res.sublist(24, 34)) s.maneuverProb];
    expect(mp.reduce(math.max) > 0.5, isTrue);
    final rp = [for (final s in res.sublist(24, 34)) s.racketProb];
    final bp = [for (final s in res.sublist(24, 34)) s.bounceProb];
    expect(rp.reduce(math.max) > bp.reduce(math.max), isTrue);
  });

  test('S8: ball stays alive through a court bounce (vy flip)', () {
    final bounce = <List<Detection>>[];
    var x = 200.0, y = 100.0;
    for (var i = 0; i < 25; i++) {
      x += 15.0;
      y += 20.0;
      bounce.add(d1(x, y, 0.9));
    }
    for (var i = 0; i < 25; i++) {
      x += 15.0;
      y -= 20.0;
      bounce.add(d1(x, y, 0.9));
    }
    final res = run(bounce);
    final aliveAfter = [for (final s in res.sublist(25)) s.hasMovingTrace];
    var maxDead = 0, cur = 0;
    for (final a in aliveAfter) {
      cur = a ? 0 : cur + 1;
      maxDead = math.max(maxDead, cur);
    }
    expect(aliveAfter.sublist(aliveAfter.length - 10).every((a) => a), isTrue);
    expect(maxDead <= 2, isTrue);
    final mp = [for (final s in res.sublist(23, 33)) s.maneuverProb];
    expect(mp.reduce(math.max) > 0.5, isTrue);
    final rp = [for (final s in res.sublist(23, 40)) s.racketProb];
    final bp = [for (final s in res.sublist(23, 40)) s.bounceProb];
    expect(bp.reduce(math.max) > rp.reduce(math.max), isTrue);
  });

  test('S9: re-acquire ball after fast serve contact', () {
    final ts = <List<Detection>>[];
    var xt = 400.0, yt = 400.0;
    for (var i = 0; i < 20; i++) {
      yt -= 20.0;
      ts.add(d1(xt, yt, 0.9));
    }
    ts.add(const []);
    var xs = xt, ys = yt;
    for (var i = 0; i < 40; i++) {
      xs += 100.0;
      ts.add(d1(xs, ys, 0.9));
    }
    final res = run(ts);
    final alivePost = [for (final s in res.sublist(22)) s.hasMovingTrace];
    expect(alivePost.any((a) => a), isTrue);
    expect(res.last.hasMovingTrace, isTrue);
  });

  test('S10: track follows ball across sparse near->far net crossing', () {
    final persp = makeImageRowPerspective(540.0);
    final mgr = BallTrackManager(fps, perspectiveScale: persp);
    final netCross = <List<Detection>>[];
    final truth = <List<double>>[];
    var xn = 150.0, yn = 460.0;
    for (var i = 0; i < 16; i++) {
      xn += 26.0;
      yn -= 11.0;
      netCross.add(d1(xn, yn, 0.9));
      truth.add([xn, yn]);
    }
    for (var i = 0; i < 7; i++) {
      xn += 15.0;
      yn -= 6.0;
      netCross.add(const []);
      truth.add([xn, yn]);
    }
    for (var i = 0; i < 28; i++) {
      xn += 15.0;
      yn -= 6.0;
      netCross.add(i % 4 == 0 ? d1(xn, yn, 0.8) : const []);
      truth.add([xn, yn]);
    }
    final errs = <double>[];
    TrackStatus? lastStatus;
    var tt = 0.0;
    for (var i = 0; i < netCross.length; i++) {
      final s = mgr.update(netCross[i], tt);
      tt += dt;
      lastStatus = s;
      if (s.position != null) {
        errs.add(math.sqrt(
            math.pow(s.position![0] - truth[i][0], 2).toDouble() +
                math.pow(s.position![1] - truth[i][1], 2).toDouble()));
      }
    }
    expect(errs.last < 60.0, isTrue);
    expect(errs.sublist(20).reduce(math.max) < 120.0, isTrue);
    expect(lastStatus!.hasMovingTrace, isTrue);
  });
}
