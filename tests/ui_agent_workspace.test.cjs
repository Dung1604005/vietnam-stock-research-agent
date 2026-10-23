// Interaction tests against the actual client asset with a minimal DOM test double.
// No browser, network, npm package or product service is started.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/stock_multi_agent/ui/agent_workspace.js'), 'utf8');

function node(dataset = {}) {
  const classes = new Set();
  return {dataset, hidden:false, disabled:false, attributes:{}, scrollTop:0, scrollLeft:0, _html:'',
    get innerHTML(){return this._html}, set innerHTML(value){this._html=value; this.scrollTop=0; this.scrollLeft=0},
    classList: {toggle(c,on) { on ? classes.add(c) : classes.delete(c); }, add(c){classes.add(c)}, remove(c){classes.delete(c)}, contains(c){return classes.has(c)}},
    setAttribute(k,v){this.attributes[k]=v;}};
}
function fixture(storage = new Map(), incoming = null) {
  const roles = ['manager','data','research','technical','fundamental','risk'];
  const tabs = ['current','history','messages','trace'];
  const buttons = roles.map(role => node({agent:role}));
  const panels = roles.map(role => {
    const panel = node({detail:role});
    panel.tabs = tabs.map(tab => node({tab}));
    panel.contents = tabs.map(content => node({content}));
    panel.replay = node();
    panel.querySelectorAll = selector => selector === '[data-tab]' ? panel.tabs : panel.contents;
    panel.querySelector = () => panel.replay;
    return panel;
  });
  const edge = node({sender:'manager',receiver:'risk'});
  let replays = 0;
  edge.querySelectorAll = () => [{beginElement(){ replays++; }}];
  const caption = {textContent:''};
  const studio = {
    querySelectorAll(selector) {
      return {'[data-agent]':buttons, '[data-detail]':panels,
        '[data-tab]':panels.flatMap(p=>p.tabs), '.task-content':panels.flatMap(p=>p.contents), '.replay-task':panels.map(p=>p.replay), '.recorded-transfer':[edge]}[selector] || [];
    },
    querySelector(selector) {
      if (selector.startsWith('[data-agent=')) return buttons.find(b=>selector.includes(`"${b.dataset.agent}"`));
      if (selector === '.working') return null;
      if (selector === '.studio-handoff-caption') return caption;
      return selector.includes('"risk"') || selector.includes('"manager"') ? edge : null;
    }
  };
  let update;
  vm.runInNewContext(source, {
    document: {body:{dataset:{defaultAgent:'manager'}}, querySelector:()=>studio, querySelectorAll:()=>[]},
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
    matchMedia:()=>({matches:false}), setTimeout:()=>1, clearTimeout:()=>{},
    channel:'studio-delta', window:{addEventListener:(name,fn)=>{update=fn}},
    DOMParser: class {parseFromString(){return {querySelector:()=>incoming?.studio}}},
  });
  return {buttons,panels,storage,edge,caption,studio,replays:()=>replays,update:markup=>update({detail:markup})};
}

test('click avatar selects its panel and tabs without Python callbacks', () => {
  const f = fixture();
  f.buttons[5].onclick();
  assert.equal(f.buttons[5].attributes['aria-pressed'], 'true');
  assert.equal(f.panels[5].hidden, false);
  assert.equal(f.panels[0].hidden, true);
  f.panels[5].tabs[2].onclick();
  assert.equal(f.panels[5].contents[2].hidden, false);
  assert.equal(f.panels[5].contents[1].hidden, true);
  assert.deepEqual(JSON.parse(f.storage.get('stock-studio-inspector')), {role:'risk',tab:'messages'});
});
test('selection survives new telemetry renders and unavailable replay is disabled', () => {
  const first = fixture();
  first.buttons[5].onclick();
  const second = fixture(first.storage);
  assert.equal(second.panels[5].hidden, false);
  assert.equal(second.panels[1].replay.disabled, true);
  assert.equal(second.panels[5].replay.disabled, false);
});
test('replay animates a recorded edge and labels it as historical', () => {
  const f = fixture();
  f.buttons[5].onclick();
  f.panels[5].replay.onclick();
  assert.equal(f.replays(), 1);
  assert.equal(f.edge.classList.contains('replaying'), true);
  assert.match(f.caption.textContent, /không phải tác vụ đang chạy/);
});

test('telemetry patches in place without losing avatar, tab or inspector scroll', () => {
  const fresh = fixture();
  fresh.panels[5].contents[2].innerHTML = 'new messages';
  const f = fixture(new Map(), fresh);
  f.buttons[5].onclick();
  f.panels[5].tabs[2].onclick();
  const avatarButton = f.buttons[5];
  const inspectorViewport = f.panels[5].contents[2];
  inspectorViewport.scrollTop = 175;
  inspectorViewport.scrollLeft = 12;
  f.update('new telemetry');
  assert.equal(f.buttons[5], avatarButton);
  assert.equal(f.panels[5].contents[2], inspectorViewport);
  assert.equal(inspectorViewport.innerHTML, 'new messages');
  assert.equal(inspectorViewport.scrollTop, 175);
  assert.equal(inspectorViewport.scrollLeft, 12);
  assert.equal(f.panels[5].hidden, false);
  assert.equal(inspectorViewport.hidden, false);
  f.update('same telemetry');
  assert.equal(inspectorViewport.scrollTop, 175);
});
