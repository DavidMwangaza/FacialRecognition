#!/usr/bin/env python3
"""Conversion d'un fichier pickle (.pkl) contenant des embeddings vers JSON.

Hypothèses de structures supportées dans le .pkl:
1. dict[str, list[float]]                --> {"embeddings": [{"id": key, "vector": [...]}, ...]}
2. dict[str, dict] avec clé 'embedding'  --> idem (cherche sous-clé 'embedding')
3. list[ (label, embedding) ]            --> converti en tableau
4. list[ dict ] avec 'name'/'id' + 'embedding' ou 'vector'

Sécurité: NE CHARGEZ QUE DES FICHIERS PICKLE DE SOURCE FIABLE.

Usage:
  python convert_embeddings_pkl_to_json.py --input data/embeddings.pkl --output android/app/src/main/assets/embeddings.json

Options:
  --precision 6    Nombre de décimales pour réduire la taille (par défaut 6)
  --max-items N    Limite du nombre d'items (debug)
  --sort           Trie par identifiant
  --pretty         JSON indenté (plus gros)
  --schema         Affiche le schéma JSON attendu et quitte

Schéma JSON produit:
{
  "version": 1,
  "dimension": 512,
  "embeddings": [
     {"id": "alice", "vector": [0.1234, ...]},
     {"id": "bob",   "vector": [...]} 
  ]
}
"""
from __future__ import annotations
import argparse, json, pickle, os, sys, math
from typing import Any, Iterable

try:
    import numpy as np  # facultatif
except ImportError:
    np = None  # type: ignore

def detect_dimension(vec: Iterable[float]) -> int:
    try:
        return len(list(vec))
    except Exception:
        return -1

def normalize_vector(v: Iterable[float], eps: float = 1e-12) -> list[float]:
    arr = list(float(x) for x in v)
    norm = math.sqrt(sum(x*x for x in arr))
    if norm < eps:
        return arr
    return [x / norm for x in arr]

def round_vector(v: Iterable[float], precision: int) -> list[float]:
    factor = 10 ** precision
    return [math.floor(x * factor + 0.5) / factor for x in v]

def extract_items(obj: Any) -> list[dict]:
    items = []
    # Case dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                # search embedding key
                if 'embedding' in v:
                    items.append({"id": str(k), "vector": v['embedding']})
                elif 'vector' in v:
                    items.append({"id": str(k), "vector": v['vector']})
                else:
                    # attempt flatten numpy or list
                    maybe = [x for x in v.values() if isinstance(x, (list, tuple)) or (np is not None and isinstance(x, np.ndarray))]
                    if maybe:
                        items.append({"id": str(k), "vector": maybe[0]})
            elif isinstance(v, (list, tuple)) or (np is not None and isinstance(v, np.ndarray)):
                items.append({"id": str(k), "vector": v})
            else:
                # ignore scalar
                pass
    elif isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                vid = el.get('id') or el.get('name') or el.get('label') or f"item_{len(items)}"
                vec = el.get('embedding') or el.get('vector')
                if vec is not None:
                    items.append({"id": str(vid), "vector": vec})
            elif isinstance(el, (list, tuple)) and len(el) == 2:
                label, vec = el
                items.append({"id": str(label), "vector": vec})
    else:
        raise ValueError("Structure pickle non supportée directement.")
    return items

def coerce_vector(v: Any) -> list[float]:
    if np is not None and isinstance(v, np.ndarray):
        return v.astype('float32').tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    raise ValueError("Format de vecteur non supporté")

def main():
    ap = argparse.ArgumentParser(description="Convertit un fichier pickle d'embeddings vers JSON")
    ap.add_argument('--input', '-i', required=False, help='Chemin du fichier .pkl')
    ap.add_argument('--output', '-o', required=False, help='Chemin de sortie .json')
    ap.add_argument('--precision', type=int, default=6, help='Décimales max par composante')
    ap.add_argument('--max-items', type=int, default=None, help='Limiter nombre items')
    ap.add_argument('--sort', action='store_true', help='Trie par id')
    ap.add_argument('--pretty', action='store_true', help='Indentation JSON')
    ap.add_argument('--normalize', action='store_true', help='Applique normalisation L2 avant arrondi')
    ap.add_argument('--schema', action='store_true', help='Affiche le schéma JSON et quitte')
    args = ap.parse_args()

    if args.schema:
        print(__doc__)
        return

    if not args.input or not args.output:
        ap.print_help()
        return

    if not os.path.isfile(args.input):
        print(f"[ERREUR] Fichier introuvable: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le pickle: {e}", file=sys.stderr)
            sys.exit(2)

    items = extract_items(data)
    if not items:
        print("[ERREUR] Aucun embedding détecté dans le pickle.", file=sys.stderr)
        sys.exit(3)

    if args.sort:
        items.sort(key=lambda x: x['id'])

    if args.max_items:
        items = items[:args.max_items]

    processed = []
    dimension = None
    for it in items:
        vec = coerce_vector(it['vector'])
        if args.normalize:
            vec = normalize_vector(vec)
        vec = round_vector(vec, args.precision)
        if dimension is None:
            dimension = detect_dimension(vec)
        processed.append({"id": it['id'], "vector": vec})

    out_obj = {
        "version": 1,
        "dimension": dimension if dimension is not None else 0,
        "count": len(processed),
        "embeddings": processed
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2 if args.pretty else None)

    print(f"[OK] Export JSON: {args.output} | items={len(processed)} dimension={dimension}")

if __name__ == '__main__':
    main()
