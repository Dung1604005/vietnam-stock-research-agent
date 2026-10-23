APP_CSS = """
<style>
    .stApp { background: #f7f8fc; color: #303448; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 7rem; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e9eaf2; }
    [data-testid="stSidebar"] .stButton button { border-radius: 12px; text-align: left; }
    [data-testid="stChatMessage"] { background: #fff; border: 1px solid #ececf4; border-radius: 18px; padding: 1.3rem; }
    [data-testid="stChatInput"] { border-radius: 18px; border-color: #d9d4ef; }
    .stButton button { border-radius: 12px; border-color: #e6e4ee; background: #fff; }
    .stButton button:hover { border-color: #9d88e4; color: #7761bd; }
    .app-kicker { color: #10a37f; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .empty-state { text-align: center; padding: 1.5rem 0 1rem; }
    .empty-state h1 { font-size: 2.1rem; margin-bottom: .5rem; }
    .empty-state p { color: #888; }
    .score-card { border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: .85rem 1rem; min-height: 92px; }
    .score-label { color: #888; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
    .score-value { font-size: 1.35rem; font-weight: 700; margin-top: .25rem; }
    .freshness { color: #888; font-size: .78rem; margin-top: .75rem; }
    .trace-row { padding: .45rem .1rem; border-bottom: 1px solid rgba(128,128,128,.14); }
    .trace-agent { color: #10a37f; font-weight: 650; }
    .route-pill { display: inline-block; border: 1px solid rgba(16,163,127,.35); border-radius: 999px; padding: .18rem .55rem; margin: .1rem; font-size: .74rem; }
    .small-disclaimer { color: #888; font-size: .76rem; border-top: 1px solid rgba(128,128,128,.15); padding-top: .8rem; margin-top: 1rem; }
    .studio-brand { display:flex; align-items:center; gap:12px; padding:8px 0 18px; font-size:1.25rem; font-weight:750; letter-spacing:-.04em; color:#33344b; }
    .studio-brand b { background:#8b71e8; color:white; border-radius:13px; width:39px; height:39px; display:grid; place-items:center; font-size:25px; }
    .studio-brand small { display:block; color:#9c9bab; font-size:10px; letter-spacing:.16em; font-weight:500; }
    .studio-page-head { display:flex; justify-content:space-between; align-items:center; margin:0 0 20px; }
    .studio-page-head h1 { font-size:1.6rem; letter-spacing:-.04em; padding:0; margin:0; color:#32344a; }
    .studio-page-head p { color:#9293a5; font-size:13px; margin:6px 0 0; }
    .studio-page-badge { font-size:11px; color:#7868a7; border:1px solid #ded8f3; border-radius:30px; padding:7px 12px; background:#f1edfc; }
    .agent-studio { background:#fff; border:1px solid #e7e7f0; border-radius:22px; overflow:hidden; box-shadow:0 8px 35px #35304605; margin-bottom:12px; }
    .studio-toolbar { padding:16px 25px; border-bottom:1px solid #f0eef6; display:flex; justify-content:space-between; color:#9693a8; font-size:10px; font-weight:650; letter-spacing:.09em; align-items:center; }
    .studio-live-dot { display:inline-block; width:6px; height:6px; background:#95b8a1; border-radius:50%; margin-right:8px; }
    .studio-divider { padding:0 12px; color:#d4d1df; }
    .studio-mode { font-size:9px; padding:4px 9px; background:#f4f2fa; border-radius:6px; color:#9182b1; }
    .studio-heading { padding:22px 26px 10px; display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .studio-heading h2 { font-size:21px; letter-spacing:-.035em; margin:0; padding:0; font-weight:700; color:#39384e; }
    .studio-heading p { margin:6px 0 0; font-size:12px; color:#9c99ad; }
    .studio-team-state { border:1px solid #e8e4f3; background:#faf8ff; color:#9985c9; border-radius:20px; padding:6px 11px; font-size:10px; white-space:nowrap; }
    .studio-floor { height:480px; position:relative; margin:8px 20px 0; overflow:hidden; border-radius:15px; background:linear-gradient(180deg,#f0eef8 0%,#f6f4fb 24%,#eeedf5 24.2%,#f5f3f9 100%); }
    .studio-floor:after { content:''; position:absolute; inset:24% 0 0; background-image:linear-gradient(30deg,transparent 49.5%,#d8d2e31f 50%,transparent 50.5%),linear-gradient(150deg,transparent 49.5%,#d8d2e31f 50%,transparent 50.5%); background-size:120px 70px; pointer-events:none; }
    .studio-window { position:absolute; width:115px; height:66px; top:18px; border:7px solid #e4e0ef; background:linear-gradient(145deg,#e1ebf7,#fafbff); border-radius:9px; box-shadow:inset 0 0 0 2px #fff9; }
    .studio-window:after { content:''; position:absolute; width:4px; height:100%; background:#e4e0ef; left:50%; }
    .window-one { left:8%; } .window-two { right:8%; }
    .studio-wall-label { position:absolute; left:50%; top:22px; transform:translateX(-50%); font-size:9px; letter-spacing:.3em; color:#aea5c3; }
    .studio-wall-label span { display:block; text-align:center; margin-top:5px; font-size:20px; font-weight:700; letter-spacing:0; color:#c8bfd9; }
    .studio-rug { position:absolute; top:140px; left:22%; width:56%; height:240px; background:#e6e0f1; border:7px solid #eeebf7; border-radius:48%; transform:rotate(-4deg); opacity:.7; }
    .studio-connections { position:absolute; inset:0; width:100%; height:100%; z-index:1; pointer-events:none; }
    .studio-connections path { fill:none; stroke:#b8a8d1; stroke-width:1.4; stroke-dasharray:5 7; opacity:.6; }
    .studio-table { position:absolute; left:50%; top:235px; transform:translate(-50%,-50%); width:205px; height:88px; border-radius:50%; background:linear-gradient(150deg,#fff,#f7f3ff); border:3px solid white; box-shadow:0 10px 0 #d3cadd,0 14px 20px #6e5b9620; display:flex; flex-direction:column; align-items:center; justify-content:center; z-index:3; }
    .studio-table>span { color:#a38dcd; font-size:21px; line-height:1; }
    .studio-table strong { font-size:10px; font-weight:600; color:#9081aa; margin:5px 0 3px; }
    .studio-table small { font-size:9px; color:#afa5bc; max-width:170px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .studio-station { position:absolute; z-index:4; width:180px; transform:translateX(-50%); text-align:center; }
    .station-0 { left:50%; top:53px; } .station-1 { left:18%; top:98px; } .station-2 { left:82%; top:98px; }
    .station-3 { left:18%; top:285px; } .station-4 { left:50%; top:285px; } .station-5 { left:82%; top:285px; }
    .station-tag { display:inline-block; font-size:9px; border-radius:6px; background:#ffffffd9; padding:3px 9px; color:#8c849f; border:1px solid #ffffff; }
    .station-scene { position:relative; height:93px; }
    .studio-avatar { position:absolute; height:103px; width:96px; left:40%; top:-8px; transform:translateX(-50%); z-index:3; filter:drop-shadow(0 4px 3px #4e426515); }
    .station-desk { position:absolute; left:13px; top:56px; width:156px; height:30px; background:linear-gradient(165deg,#fff,#ebe7f3); transform:skewY(-5deg); border-radius:7px; box-shadow:0 6px 0 #d1c8df,0 12px 15px #60517310; z-index:4; }
    .station-desk:before,.station-desk:after { content:''; position:absolute; top:32px; height:18px; width:7px; background:#c9c0d7; border-radius:0 0 4px 4px; }.station-desk:before { left:10px; }.station-desk:after { right:10px; }
    .station-monitor { position:absolute; top:21px; right:24px; height:39px; width:46px; border:4px solid #cec6dd; background:#fefeff; border-radius:5px; z-index:5; transform:skewY(5deg); box-shadow:3px 3px 0 #bdb2ce; }
    .station-monitor:after { content:''; position:absolute; left:17px; top:35px; width:5px; height:9px; background:#bdb2ce; }
    .station-monitor span,.station-monitor i,.station-monitor b { position:absolute; height:3px; left:5px; background:var(--agent); border-radius:2px; opacity:.65; }.station-monitor span { top:7px; width:21px; }.station-monitor i { top:14px; width:15px; }.station-monitor b { top:21px; width:26px; }
    .station-name { margin-top:12px; font-size:12px; font-weight:700; color:#5b526d; }.station-name span { margin-left:4px; font-size:9px; color:#a69cb5; font-weight:400; }
    .station-status { font-size:9px; color:#aaa1b5; margin-top:4px; }.station-status i { display:inline-block; width:4px; height:4px; background:#c8c0d2; border-radius:50%; margin-right:5px; vertical-align:middle; }
    .state-started { color:#7e66b1; }.state-started i { background:#9673db; box-shadow:0 0 0 3px #9673db20; }.state-succeeded i { background:#75b78c; }.state-failed i { background:#d87e8b; }.state-partial i,.state-fallback i { background:#d5ad62; }
    .selected .station-tag { border-color:var(--agent); color:var(--agent); }.working .studio-avatar { animation:studio-bob 1.8s ease-in-out infinite; }.working .station-monitor { box-shadow:0 0 14px var(--agent); }
    .studio-plant { position:absolute; color:#a9bea9; font-size:54px; z-index:2; line-height:1; }.studio-plant:after { content:''; display:block; background:#ddd5e1; width:28px; height:25px; border-radius:0 0 10px 10px; margin:-8px auto 0; }.plant-left { bottom:28px; left:3%; }.plant-right { top:155px; right:3%; }
    .studio-floor-note { position:absolute; bottom:12px; left:50%; transform:translateX(-50%); color:#b1a7be; font-size:9px; z-index:5; white-space:nowrap; }.studio-floor-note span { padding:0 7px; }
    .studio-wall-label span { display:none; }.studio-table { top:253px; }
    .station-name { font-size:13px; }.station-status { font-size:10px; }.station-tag { font-size:10px; }
    .studio-inspector { margin:18px 22px 0; padding:10px 16px; display:flex; align-items:center; gap:12px; border:1px solid #eeeaf6; border-radius:12px; background:#fcfbfe; }.studio-inspector img { width:42px; height:45px; }.studio-inspector strong { font-size:12px; color:#615470; }.studio-inspector strong span { color:#aaa0b7; font-weight:400; }.studio-inspector p { font-size:11px; color:#a399b0; margin:4px 0 0; }.studio-inspector-status { margin-left:auto; color:#9b8cad; font-size:10px; white-space:nowrap; }
    .studio-footnote { text-align:center; padding:13px 10px 16px; color:#b1a8bc; font-size:9px; }
    @keyframes studio-bob { 0%,100% { margin-top:0; } 50% { margin-top:-5px; } }
    @media(prefers-reduced-motion:reduce) { .working .studio-avatar { animation:none; } }
    @media(max-width:700px) { .block-container { padding-top:1rem; }.studio-heading { padding:17px 15px 5px; }.studio-heading h2 { font-size:17px; }.studio-team-state { display:none; }.studio-floor { margin:8px 7px 0; }.studio-station { width:125px; }.station-desk { left:6px; width:112px; }.station-monitor { right:8px; }.station-1,.station-3 { left:17%; }.station-2,.station-5 { left:83%; }.studio-table { width:155px; height:76px; }.station-name span { display:block; margin-left:0; }.studio-inspector { margin:13px 10px 0; padding:8px; }.studio-inspector-status { display:none; }.studio-window { width:70px; }.studio-page-badge { display:none; }.studio-footnote { font-size:8px; }.studio-floor-note { bottom:8px; } }
</style>
"""

