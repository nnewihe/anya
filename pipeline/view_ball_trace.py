"""
view_ball_trace.py
=================
Build a self-contained interactive HTML viewer of the cached ball telemetry, so
you can eyeball whether what's being tracked in each near-side rally is a live
ball — and, crucially, whether the detections that continue AFTER the ground-
truth point end (the tail that inflates the energy-bar late bias) are a real
moving ball or a stationary/dead one.

Reads <clip>/energy_telemetry_cache.json (no re-detection). Draws each rally's
ball path on the 960x540 court, colours in-rally vs post-end frames differently,
animates playback, and reports a "tail spread" (how far the post-end ball moves)
— low spread == stationary/dead ball that shouldn't count as live.

Usage:
    python pipeline/view_ball_trace.py                 # all cached clips
    python pipeline/view_ball_trace.py --clips 36 58
    python pipeline/view_ball_trace.py --out /tmp/trace.html
"""

import os
import json
import math
import argparse

ANALYSIS = (960, 540)
TELEMETRY_CACHE = "energy_telemetry_cache.json"


def _compact_rally(r):
    fr = r["frames"]
    start, end, span_end = r["start"], r["end"], r["span_end"]
    pts = []
    tail_xy = []
    in_ball = in_n = tail_ball = tail_n = 0
    for f in range(start, span_end + 1):
        cell = fr.get(str(f))
        b = cell["ball"] if cell else None
        after = f > end
        if after:
            tail_n += 1
        else:
            in_n += 1
        if b:
            pts.append([f - start, round(b[0], 1), round(b[1], 1)])
            if after:
                tail_ball += 1
                tail_xy.append(b)
            else:
                in_ball += 1
    # tail spread: max pairwise distance of post-end ball positions (px)
    spread = 0.0
    if len(tail_xy) >= 2:
        xs = [p[0] for p in tail_xy]; ys = [p[1] for p in tail_xy]
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    # tail coherence: median consecutive-frame displacement of post-end detections
    tj = []
    for i in range(1, len(pts)):
        (o0, x0, y0), (o1, x1, y1) = pts[i - 1], pts[i]
        if o0 > (end - start) and o1 - o0 == 1:
            tj.append(math.hypot(x1 - x0, y1 - y0))
    tj.sort()
    tail_jump = round(tj[len(tj) // 2]) if tj else 0
    return {
        "start": start, "end": end, "span_end": span_end,
        "end_off": end - start, "span_off": span_end - start,
        "pts": pts,
        "in_pct": round(100 * in_ball / in_n) if in_n else 0,
        "tail_pct": round(100 * tail_ball / tail_n) if tail_n else 0,
        "tail_spread": round(spread),
        "tail_jump": tail_jump,
    }


def build_data(clip_dirs):
    data = {}
    for c in clip_dirs:
        path = os.path.join(c, TELEMETRY_CACHE)
        if not os.path.isfile(path):
            continue
        try:
            tel = json.load(open(path))
        except Exception as e:
            print(f"[skip] {os.path.basename(c)}: {e}")
            continue
        data[os.path.basename(c)] = {
            "fps": round(tel["fps"], 3),
            "corners": tel["corners"],
            "rallies": [_compact_rally(r) for r in tel["rallies"]],
        }
        print(f"[load] {os.path.basename(c)}: {len(tel['rallies'])} rallies")
    return data


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Ball trace viewer</title>
<style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#ddd}
  header{padding:10px 14px;background:#1b1b1b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  select,button{font-size:14px;padding:5px 8px;background:#262626;color:#eee;border:1px solid #3a3a3a;border-radius:6px}
  button{cursor:pointer}
  #wrap{display:flex;gap:16px;padding:14px;flex-wrap:wrap}
  canvas{background:#0c2a12;border:1px solid #333;border-radius:6px;max-width:100%}
  #stats{font-size:14px;line-height:1.9}
  .tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600}
  .live{background:#12351a;color:#5dcaa5}.dead{background:#3a1414;color:#e88}
  #slider{width:520px}
  label{font-size:13px;color:#aaa}
  b{color:#fff}
</style></head><body>
<header>
  <span><label>clip</label> <select id="clip"></select></span>
  <span><label>rally</label> <select id="rally"></select></span>
  <button id="play">▶ play</button>
  <input type="range" id="slider" min="0" max="100" value="0">
  <label><input type="checkbox" id="full" checked> full trace</label>
  <label><input type="checkbox" id="tailonly"> tail only</label>
</header>
<div id="wrap">
  <canvas id="cv" width="960" height="540"></canvas>
  <div id="stats"></div>
</div>
<script>
const DATA = __DATA__;
const W=960,H=540;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const $clip=document.getElementById('clip'), $rally=document.getElementById('rally');
const $slider=document.getElementById('slider'), $play=document.getElementById('play');
const $full=document.getElementById('full'), $tail=document.getElementById('tailonly'), $stats=document.getElementById('stats');
let clip=null, rally=null, playing=false, raf=null;

for(const k of Object.keys(DATA)){const o=document.createElement('option');o.value=k;o.textContent='clip '+k+' ('+DATA[k].rallies.length+')';$clip.appendChild(o);}

function setClip(k){clip=k;$rally.innerHTML='';DATA[k].rallies.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent='#'+(i+1)+'  ['+r.start+'-'+r.end+']';$rally.appendChild(o);});setRally(0);}
function setRally(i){rally=DATA[clip].rallies[i];$slider.max=rally.span_off;$slider.value=rally.span_off;draw();stats();}

function courtPath(){const c=DATA[clip].corners;ctx.beginPath();ctx.moveTo(c[0][0],c[0][1]);for(let i=1;i<4;i++)ctx.lineTo(c[i][0],c[i][1]);ctx.closePath();}
function col(off){ // blue(in-rally early)->cyan(late), orange->red for post-end tail
  if(off<=rally.end_off){const t=off/Math.max(1,rally.end_off);return `rgb(${40+0*t},${140+80*t},${255-60*t})`;}
  const t=(off-rally.end_off)/Math.max(1,rally.span_off-rally.end_off);return `rgb(255,${150-120*t|0},${40})`;
}
function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(255,255,255,.25)';ctx.lineWidth=1.5;courtPath();ctx.stroke();
  const cur=+$slider.value;
  const pts=rally.pts.filter(p=> $tail.checked ? p[0]>rally.end_off : true);
  if($full.checked){
    for(const [off,x,y] of pts){ if(off>cur && !$tail.checked) continue;
      ctx.fillStyle=col(off);ctx.globalAlpha=.85;ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill();}
    ctx.globalAlpha=1;
  }
  // current ball + short trail
  let cx=null,cy=null;
  for(let i=pts.length-1;i>=0;i--){ if(pts[i][0]<=cur){cx=pts[i][1];cy=pts[i][2];break;} }
  if(cx!==null){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,8,0,7);ctx.stroke();}
  // marker line: is current frame past GT end?
  const post=cur>rally.end_off;
  ctx.fillStyle=post?'#e88':'#5dcaa5';ctx.font='14px sans-serif';
  ctx.fillText((post?'POST-END (dead?)':'IN RALLY')+'  frame off '+cur+'/'+rally.span_off, 12, 22);
}
function stats(){
  const stationary = rally.tail_spread<60;
  $stats.innerHTML =
    '<div>clip <b>'+clip+'</b> rally <b>'+($rally.selectedIndex+1)+'</b> — frames '+rally.start+'–'+rally.end+' (+'+(rally.span_off-rally.end_off)+' tail)</div>'+
    '<div>fps '+DATA[clip].fps+'</div><br>'+
    '<div>in-rally ball coverage: <b>'+rally.in_pct+'%</b></div>'+
    '<div>post-end ball coverage: <b>'+rally.tail_pct+'%</b></div>'+
    '<div>post-end ball spread: <b>'+rally.tail_spread+' px</b> '+
      '<span class="tag '+(stationary?'dead':'live')+'">'+(stationary?'stationary cluster':'moves across frame')+'</span></div>'+
    '<div>post-end frame-to-frame jump (median): <b>'+rally.tail_jump+' px/f</b></div>'+
    '<br><div style="color:#888;font-size:12px;max-width:340px">Blue→cyan = in-rally, orange→red = post-end tail. Watch the tail: if it is a single ball being retrieved/knocked around it moves like the rally ball — the trace cannot tell live-rally from between-points on ball motion alone, which is why the energy bar ends ~4s late.</div>';
}
function loop(){ if(!playing)return; let v=+$slider.value+2; if(v>rally.span_off)v=0; $slider.value=v; draw();
  raf=requestAnimationFrame(()=>setTimeout(loop,33)); }
$play.onclick=()=>{playing=!playing;$play.textContent=playing?'❚❚ pause':'▶ play'; if(playing)loop();};
$slider.oninput=draw; $full.onchange=draw; $tail.onchange=()=>{draw();};
$clip.onchange=e=>setClip(e.target.value); $rally.onchange=e=>setRally(+e.target.value);
setClip(Object.keys(DATA)[0]);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Interactive ball-trace viewer from cached telemetry")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="/Volumes/Anya/Data/ball_trace_viewer.html")
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if os.path.isfile(os.path.join(args.data_root, d, TELEMETRY_CACHE))]

    data = build_data(clip_dirs)
    if not data:
        raise SystemExit("No telemetry caches found.")

    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(args.out, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[done] {len(data)} clips -> {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
