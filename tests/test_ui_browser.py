import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
UI = ROOT / "boardforge" / "static" / "index.html"


def _chrome() -> str | None:
    candidates = (
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
    return next((value for value in candidates if value and Path(value).exists()), None)


MOCK_FETCH = r"""
window.__calls=[];window.__agentCalls=0;window.__profileADelay=0;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const profileA={skills:['Python','FastAPI'],role_signals:['Backend'],target_roles:[],domains:['Backend / APIs'],education:['Example IIT'],projects:['Built an API'],experience:[],reviewed:false,resume_filename:'a.pdf',evidence:[
  {kind:'skill',value:'Python',snippet:'Python'},
  {kind:'role',value:'Backend',snippet:'Built a FastAPI backend'},
  {kind:'education',value:'Example IIT',snippet:'Example IIT'},
  {kind:'project',value:'Built an API',snippet:'Built an API'}
]};
const response=(status,data,delay=0)=>new Promise(resolve=>setTimeout(()=>resolve({status,ok:status>=200&&status<300,json:async()=>data}),delay));
window.fetch=(path,options={})=>{
  const headers=new Headers(options.headers||{}),auth=headers.get('Authorization')||'',method=options.method||'GET';
  window.__calls.push({path:String(path),method,auth,body:typeof options.body==='string'?options.body:'<binary>'});
  if(path==='/health')return response(200,{version:'test'});
  if(path==='/session'){
    if(auth==='Bearer token-a')return response(200,{profile_id:'profile-a',auth_required:true});
    if(auth==='Bearer token-b')return response(200,{profile_id:'profile-b',auth_required:true});
    return response(401,{detail:'valid invite token required'});
  }
  if(String(path).startsWith('/candidate-profile/profile-a')){
    if(method==='PUT')return response(200,{profile:{...profileA,target_roles:['Backend'],reviewed:true}});
    if(method==='DELETE')return response(200,{deleted:true,profile_id:'profile-a'},20);
    return response(200,{profile:{...profileA,resume_filename:'A Private.pdf'}},window.__profileADelay);
  }
  if(String(path).startsWith('/candidate-profile/profile-b'))return response(200,{profile:null});
  if(String(path).startsWith('/resume'))return response(200,{profile:{...profileA,resume_filename:'new.pdf'}},30);
  if(path==='/agent'){
    window.__agentCalls++;
    if(window.__agentCalls===2)return response(500,{detail:'mock agent unavailable'},20);
    return response(200,{message:'One grounded recommendation.',agent:{stopped_reason:'sufficient_results'},trace:[{tool:'search_indexed_jobs',ok:true,arguments:{query:'backend'}}],matches:[
      {job:{stable_id:'test:jobs:1',title:'Backend <script> Engineer',company:'Example',location:'India',apply_url:'https://jobs.example/apply'},score:.91,status:'recommended',reasons:['FastAPI match'],risks:[],evidence:[{detail:'Build production FastAPI services.'}]},
      {job:{stable_id:'test:jobs:2',title:'Unsafe Link',company:'Example',location:'India',apply_url:'javascript:alert(1)'},score:.4,status:'needs_verification',reasons:[],risks:['Eligibility unknown'],evidence:[]}
    ]},60);
  }
  if(path==='/feedback')return response(200,{message:'recorded'},30);
  if(path==='/preferences')return response(200,{message:'saved',preferences:{}},20);
  return response(404,{detail:'unmocked '+path});
};
window.confirm=()=>true;window.alert=message=>{window.__lastAlert=message};
"""


HARNESS = r"""
(async()=>{
  const waitFor=async(predicate,label)=>{for(let i=0;i<100;i++){if(predicate())return;await sleep(10)}throw new Error('timeout: '+label)};
  const assert=(condition,message)=>{if(!condition)throw new Error(message)};
  try{
    await waitFor(()=>!document.getElementById('accessPanel').hidden,'initial access panel');
    document.getElementById('inviteToken').value='token-a';
    document.getElementById('inviteForm').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
    await waitFor(()=>document.getElementById('profileCard').classList.contains('show'),'profile restoration');
    assert(document.getElementById('profileFile').textContent==='A Private.pdf','restored profile not rendered');
    assert([...document.querySelectorAll('#profileEvidence .evidence-item')].some(x=>x.textContent.includes('education')),'representative evidence missing');

    document.getElementById('targetRoles').value='Backend';
    document.getElementById('reviewProfile').click();
    await waitFor(()=>document.getElementById('profileStatus').textContent.includes('confirmed'),'profile confirmation');

    const upload=uploadResume(new File(['resume'],'new.pdf',{type:'application/pdf'}));
    assert(document.getElementById('chooseResume').disabled,'upload loading state missing');
    assert(document.getElementById('resumeStatus').textContent.includes('Reading'),'upload status missing');
    await upload;
    assert(document.getElementById('profileFile').textContent==='new.pdf','upload response not rendered');

    document.getElementById('query').value='backend';
    document.getElementById('searchBtn').click();document.getElementById('searchBtn').click();
    assert(document.getElementById('searchBtn').disabled,'search loading state missing');
    await waitFor(()=>document.querySelectorAll('.job').length===2,'agent results');
    assert(window.__agentCalls===1,'concurrent duplicate search was sent');
    assert(!document.querySelector('#results script'),'job title created executable markup');
    assert(document.querySelectorAll('a[href^="https://jobs.example"]').length===1,'valid application link missing');
    assert(![...document.querySelectorAll('.job')][1].querySelector('a'),'unsafe application link rendered');
    assert(document.querySelector('.source-evidence').textContent.includes('FastAPI'),'evidence not rendered');
    assert(document.getElementById('activitySteps').textContent.includes('Searched the indexed jobs'),'agent trace not rendered');

    const saved=document.querySelector('.feedback[data-v="save"]');saved.click();
    assert([...saved.closest('.job').querySelectorAll('.feedback')].every(x=>x.disabled),'concurrent feedback controls not locked');
    await waitFor(()=>saved.textContent==='Saved','save feedback');
    const feedbackCall=window.__calls.find(x=>x.path==='/feedback');
    assert(JSON.parse(feedbackCall.body).job_id==='test:jobs:1','feedback stable id incorrect');

    document.getElementById('query').value='error case';document.getElementById('searchBtn').click();
    await waitFor(()=>document.getElementById('searchStatus').textContent.includes('previous results retained'),'error state');
    assert(document.querySelectorAll('.job').length===2,'error erased previous results');

    window.__profileADelay=100;
    const oldConnect=connect('token-a');await sleep(10);const newConnect=connect('token-b');
    await Promise.all([oldConnect,newConnect]);await sleep(110);
    assert(profileId==='profile-b','latest authenticated user did not win');
    assert(document.getElementById('profileFile').textContent!=='A Private.pdf','stale profile leaked across users');
    assert(!document.getElementById('profileCard').classList.contains('show'),'prior user profile remained visible');

    await connect('token-a');
    await waitFor(()=>document.getElementById('profileCard').classList.contains('show'),'profile before deletion');
    document.getElementById('deleteProfile').click();
    assert(document.getElementById('deleteProfile').disabled,'delete loading state missing');
    await waitFor(()=>!document.getElementById('profileCard').classList.contains('show'),'profile deletion');
    assert(document.getElementById('searchStatus').textContent.includes('Profile deleted'),'delete confirmation missing');

    const protectedCalls=window.__calls.filter(x=>x.path!=='/health'&&x.path!=='/session'&&x.path!=='/candidate-profile/profile-b');
    assert(protectedCalls.every(x=>x.auth==='Bearer token-a'),'protected request omitted invite token');
    document.body.dataset.auditResult='pass';
    document.body.dataset.auditCalls=String(window.__calls.length);
  }catch(error){document.body.dataset.auditResult='fail: '+error.message}
})();
"""


@pytest.mark.skipif(_chrome() is None, reason="Chrome or Edge is required for deterministic browser validation")
def test_editorial_ui_behaviour_with_mocked_api(tmp_path):
    html = UI.read_text(encoding="utf-8")
    html = html.replace("<script>", f"<script>{MOCK_FETCH}</script><script>", 1)
    html = html.replace("</body>", f"<script>{HARNESS}</script></body>")
    harness = tmp_path / "boardforge-ui-audit.html"
    harness.write_text(html, encoding="utf-8")
    profile = tmp_path / "chrome-profile"
    command = [
        _chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--host-resolver-rules=MAP * 0.0.0.0", f"--user-data-dir={profile}",
        "--virtual-time-budget=10000", "--dump-dom", harness.as_uri(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'data-audit-result="pass"' in completed.stdout, completed.stdout[-3000:]
