// Utility for persisting File objects in IndexedDB so uploads survive page navigation
// (Metadata is stored in localStorage separately.)
// This is intentionally lightweight to avoid adding a dependency.

const DB_NAME = 'bl_uploads_db';
const STORE = 'files';
let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE);
      }
    };
    req.onerror = () => reject(req.error || new Error('IndexedDB open error'));
    req.onsuccess = () => resolve(req.result);
  });
  return dbPromise;
}

export async function persistFile(id: string, file: File): Promise<void> {
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.onabort = () => reject(tx.error || new Error('tx aborted'));
      tx.onerror = () => reject(tx.error || new Error('tx error'));
      tx.oncomplete = () => resolve();
      tx.objectStore(STORE).put(file, id);
    });
  } catch (e) {
    // swallow errors (private / incognito modes may block IDB)
    // eslint-disable-next-line no-console
    console.warn('[uploadPersistence] persistFile failed', e);
  }
}

export async function loadFile(id: string): Promise<File | undefined> {
  try {
    const db = await openDB();
    return await new Promise<File | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      tx.onabort = () => reject(tx.error || new Error('tx abort'));
      tx.onerror = () => reject(tx.error || new Error('tx error'));
      const req = tx.objectStore(STORE).get(id);
      req.onsuccess = () => resolve(req.result as File | undefined);
      req.onerror = () => reject(req.error || new Error('get error'));
    });
  } catch {
    return undefined;
  }
}

export async function removeFile(id: string): Promise<void> {
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.onabort = () => reject(tx.error || new Error('tx abort'));
      tx.onerror = () => reject(tx.error || new Error('tx error'));
      tx.oncomplete = () => resolve();
      tx.objectStore(STORE).delete(id);
    });
  } catch {}
}
