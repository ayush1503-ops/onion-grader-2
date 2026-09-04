/* ============================================================
 sw.js - service worker for the OFFLINE onion grader
 First load (needs the files once): caches everything.
 After that: the app works with NO INTERNET at all.
 ============================================================ */
const CACHE = "onion-grader-offline-v3";
const ASSETS = [
  "./",
  "./index.html",
  "./opencv.js",
  "./manifest.webmanifest",
  "./icon-192.png",
  "./icon-512.png",
  "./hero_onion.jpg",
  "./three.min.js"
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

/* network first, cache fallback (so updates arrive when online,
   and everything still works with zero connectivity) */
self.addEventListener("fetch", (e) => {
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
