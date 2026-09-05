// Extract and exercise the pure-logic helpers from the upload flow.
const fs=require('fs');
const html=fs.readFileSync('app_page.html','utf8');
const js=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join("\n");

// pull out the two helpers we want to verify in isolation
function grab(name){
  const i=js.indexOf("function "+name+"(");
  if(i<0) throw new Error("not found: "+name);
  let d=0,started=false;
  for(let j=i;j<js.length;j++){
    if(js[j]==='{'){d++;started=true;}
    else if(js[j]==='}'){d--;if(started&&d===0)return js.slice(i,j+1);}
  }
}
eval(grab("uploadName"));
eval(grab("isHeic"));

let fails=0;
function eq(got,want,label){
  const ok=got===want; if(!ok)fails++;
  console.log(`  ${ok?"PASS":"FAIL"}  ${label.padEnd(46)} -> ${JSON.stringify(got)}${ok?"":" (want "+JSON.stringify(want)+")"}`);
}

console.log("uploadName(): server must always receive a .jpg name");
eq(uploadName({name:"IMG_0421.HEIC"}), "IMG_0421.jpg", "iPhone HEIC name");
eq(uploadName({name:"IMG_0421.heic"}), "IMG_0421.jpg", "lowercase .heic");
eq(uploadName({name:"photo.JPG"}),     "photo.jpg",    "uppercase .JPG");
eq(uploadName({name:"Screenshot.png"}),"Screenshot.jpg","png screenshot");
eq(uploadName({name:""}),              "photo.jpg",    "empty name (iOS blob)");
eq(uploadName({}),                     "photo.jpg",    "no name property");
eq(uploadName({name:"blob"}),          "blob.jpg",     "android 'blob'");
eq(uploadName({name:"../../etc/passwd"}),".._.._etc_passwd.jpg","path traversal stripped");
eq(uploadName({name:"my onion.jpeg"}), "my onion.jpg", "space + .jpeg");

console.log("\nisHeic(): detect iPhone photos by type OR name");
eq(isHeic({type:"image/heic",name:"a"}), true,  "type image/heic");
eq(isHeic({type:"image/heif",name:"a"}), true,  "type image/heif");
eq(isHeic({type:"",name:"IMG.HEIC"}),    true,  "name .HEIC, no type");
eq(isHeic({type:"image/jpeg",name:"a.jpg"}), false, "plain jpeg");
eq(isHeic({}),                           false, "empty object");

console.log(fails? `\n${fails} FAILED` : "\nALL JS LOGIC CHECKS PASSED");
process.exit(fails?1:0);
