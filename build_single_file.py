#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Junta varios modulos Python (pa_*.py, e opcionalmente pb_*.py / pc_*.py) em UM arquivo
unico, como exige o enunciado. Nao altera os modulos originais.

USO
  python build_single_file.py                                   # so a Pessoa A -> pa_unificado.py
  python build_single_file.py --header cabecalho.txt --out G01.py
  python build_single_file.py --modules "pa_*.py" "pb_*.py" "pc_*.py" --out G01.py --header cabecalho.txt

COMO FUNCIONA (tudo por AST, nao por busca de texto)
  1. Descobre as dependencias entre os modulos a partir dos imports e ordena (topologico).
     O modulo de entrada (--entry, padrao pa_gui_app.py) fica por ultimo.
  2. Remove os imports internos (from pa_x import ...; import pa_x), inclusive os feitos
     dentro de funcoes (viram 'pass'), e troca 'pa_x.nome' por 'nome'.
  3. Sobe para o topo os imports de bibliotecas externas (sem duplicar); os imports
     preguicosos dentro de funcoes (torch, nibabel...) continuam onde estao.
  4. Remove os blocos  if __name__ == "__main__"  dos modulos que nao sao a entrada.
  5. Para e avisa se dois modulos definem o MESMO nome (a fusao sobrescreveria um deles).
  6. Escreve o cabecalho com os integrantes (o enunciado exige nome/matricula no topo).

LIMITES (o script avisa, nao esconde)
  - Codigo no nivel do modulo roda na importacao. Se um modulo executa um pipeline
    inteiro ao ser importado (ex.: um script solto), embrulhe em funcoes antes de fundir.
  - So um modulo e a entrada: a linha de comando de outros modulos (ex.: um 'main' de
    treino) deixa de existir no arquivo unico.
"""
import argparse
import ast
import glob
import os
import re
import sys

HEADER_TEMPLATE = '''# -*- coding: utf-8 -*-
# =============================================================================
# PUC Minas - ICEI - Ciencia da Computacao - Processamento e Analise de Imagens
# Trabalho Pratico: Classificacao da resposta ao tratamento do cancer de mama
# Grupo: Gnn
# Integrantes (nome, matricula, curso, campus):
#   - NOME COMPLETO 1, MATRICULA, CURSO, CAMPUS
#   - NOME COMPLETO 2, MATRICULA, CURSO, CAMPUS
#   - NOME COMPLETO 3, MATRICULA, CURSO, CAMPUS
# =============================================================================
'''


class BundleError(Exception):
    pass


class Mod:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            self.text = f.read()
        self.lines = self.text.split("\n")
        try:
            self.tree = ast.parse(self.text, path)
        except SyntaxError as e:
            raise BundleError(f"{path}: erro de sintaxe: {e}")
        self.deps = set()
        self.hoisted = []          # imports externos do topo (texto normalizado)
        self.defined = set()       # nomes definidos no nivel do modulo
        self.side_effects = []     # linhas com codigo executado na importacao
        self.doc = ""              # docstring do modulo (vira comentario)
        self.body_lines = []       # resultado final


def _bound_names(node):
    out = []
    for a in node.names:
        out.append(a.asname or a.name.split(".")[0])
    return out


def _top_defs(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
    return names


def _is_main_guard(n):
    if not isinstance(n, ast.If) or not isinstance(n.test, ast.Compare):
        return False
    t = n.test
    return (isinstance(t.left, ast.Name) and t.left.id == "__name__" and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__")


def process(mod, internal, is_entry):
    tree, lines = mod.tree, mod.lines
    top_ids = {id(n) for n in tree.body}
    remove, replace, edits = set(), {}, []

    def mark(node):
        rng = range(node.lineno - 1, node.end_lineno)
        if id(node) in top_ids:
            remove.update(rng)
        else:
            first = node.lineno - 1
            indent = re.match(r"\s*", lines[first]).group(0)
            replace[first] = indent + "pass  # import interno removido na fusao"
            remove.update(range(first + 1, node.end_lineno))

    # docstring do modulo -> comentario
    if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        d = tree.body[0]
        mod.doc = d.value.value.strip()
        remove.update(range(d.lineno - 1, d.end_lineno))

    # imports
    attr_name_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in internal:
            attr_name_ids.add(id(node.value))
            edits.append(node)
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
            inn = [r for r in roots if r in internal]
            if inn:
                if len(inn) != len(roots):
                    raise BundleError(f"{mod.path}:{node.lineno}: import misto (interno + externo) na mesma linha")
                if any(a.asname or "." in a.name for a in node.names):
                    raise BundleError(f"{mod.path}:{node.lineno}: 'import x as y' de modulo interno nao e suportado")
                mod.deps.update(inn)
                mark(node)
            elif id(node) in top_ids:
                mod.hoisted.append(ast.unparse(node))
                mark(node)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise BundleError(f"{mod.path}:{node.lineno}: import relativo nao e suportado")
            root = (node.module or "").split(".")[0]
            if root in internal:
                if any(a.name == "*" for a in node.names):
                    raise BundleError(f"{mod.path}:{node.lineno}: 'import *' nao e suportado")
                if any(a.asname and a.asname != a.name for a in node.names):
                    raise BundleError(f"{mod.path}:{node.lineno}: 'from x import a as b' interno nao e suportado")
                mod.deps.add(root)
                mark(node)
            elif id(node) in top_ids:
                mod.hoisted.append(ast.unparse(node))
                mark(node)

    # modulo usado como objeto (fora de 'modulo.atributo') nao da para fundir
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in internal and id(node) not in attr_name_ids:
            raise BundleError(f"{mod.path}:{node.lineno}: o modulo '{node.id}' e usado como objeto "
                              f"(so 'modulo.nome' e suportado)")

    # blocos __main__ dos modulos que nao sao a entrada
    for n in tree.body:
        if _is_main_guard(n) and not is_entry:
            remove.update(range(n.lineno - 1, n.end_lineno))

    mod.defined = _top_defs(tree)
    for n in tree.body:
        if isinstance(n, (ast.For, ast.While, ast.With)) or (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)):
            if n.lineno - 1 not in remove:
                mod.side_effects.append(n.lineno)

    # aplica: 'modulo.' -> '' (em bytes: as colunas do ast sao offsets UTF-8)
    by_line = {}
    for a in edits:
        l0 = a.value.lineno - 1
        if l0 in remove or l0 in replace:
            continue
        by_line.setdefault(l0, []).append((a.value.col_offset, a.value.end_col_offset))
    out = []
    for i, line in enumerate(lines):
        if i in replace:
            out.append(replace[i])
            continue
        if i in remove:
            continue
        if i in by_line:
            b = line.encode("utf-8")
            for s, e in sorted(by_line[i], reverse=True):
                if b[e:e + 1] != b".":
                    raise BundleError(f"{mod.path}:{i + 1}: formato inesperado em '{mod.name}.<nome>'")
                b = b[:s] + b[e + 1:]
            line = b.decode("utf-8")
        out.append(line)
    # remove linhas em branco em excesso
    text = re.sub(r"\n{4,}", "\n\n\n", "\n".join(out)).strip("\n")
    mod.body_lines = text


def order_modules(mods, entry_name):
    by_name = {m.name: m for m in mods}
    order, state = [], {}

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise BundleError("dependencia circular: " + " -> ".join(path + [n]))
        state[n] = 1
        for d in sorted(by_name[n].deps):
            if d == entry_name:
                raise BundleError(f"o modulo '{n}' depende do modulo de entrada '{entry_name}'")
            visit(d, path + [n])
        state[n] = 2
        order.append(n)

    for m in mods:
        if m.name != entry_name:
            visit(m.name, [])
    order.append(entry_name)
    return [by_name[n] for n in order]


def build(paths, entry, header_text):
    mods = [Mod(p) for p in paths]
    names = {m.name for m in mods}
    entry_name = os.path.splitext(os.path.basename(entry))[0]
    if entry_name not in names:
        raise BundleError(f"modulo de entrada '{entry}' nao esta na lista de modulos")
    for m in mods:
        process(m, names, m.name == entry_name)
    ordered = order_modules(mods, entry_name)

    # imports externos do topo, sem duplicar
    seen, hoisted, bound = set(), [], set()
    for m in ordered:
        for h in m.hoisted:
            if h not in seen:
                seen.add(h)
                hoisted.append(h)
                bound.update(_bound_names(ast.parse(h).body[0]))

    # colisoes de nomes entre modulos (e entre definicoes e imports)
    owner, problems = {}, []
    for m in ordered:
        for n in m.defined:
            if n in owner:
                problems.append(f"'{n}' definido em {owner[n]} e em {m.name}")
            owner[n] = m.name
            if n in bound:
                problems.append(f"'{n}' definido em {m.name} e tambem importado de biblioteca")
    if problems:
        raise BundleError("colisao de nomes (renomeie em um dos modulos):\n  " + "\n  ".join(sorted(problems)))

    parts = [header_text.rstrip("\n") + "\n", "\n".join(hoisted) + "\n"]
    for i, m in enumerate(ordered, 1):
        banner = ["# " + "=" * 77, f"# [{i}/{len(ordered)}] {m.name}.py"]
        if m.doc:
            banner += ["#"] + [("# " + l).rstrip() for l in m.doc.split("\n")]
        banner.append("# " + "=" * 77)
        parts.append("\n" + "\n".join(banner) + "\n\n" + m.body_lines + "\n")
    return "".join(parts), ordered


def check(src, out_path):
    """Verificacoes sobre o arquivo gerado. Retorna lista de erros (vazia = ok)."""
    errs = []
    try:
        compile(src, out_path, "exec")
    except SyntaxError as e:
        return [f"nao compila: {e}"]
    for i, l in enumerate(src.split("\n"), 1):
        if re.match(r"\s*(from|import)\s+(pa|pb|pc)_\w+", l):
            errs.append(f"linha {i}: sobrou import interno: {l.strip()}")
    try:
        from pyflakes.api import check as pf_check
        from pyflakes.reporter import Reporter
        import io
        buf = io.StringIO()
        pf_check(src, out_path, Reporter(buf, buf))
        for l in buf.getvalue().splitlines():
            if "undefined name" in l or "redefinition of unused" in l and "from line" not in l:
                errs.append(l)
    except ImportError:
        print("  (pyflakes nao instalado: pulei a checagem de nomes indefinidos; "
              "instale com 'pip install pyflakes')")
    return errs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Junta os modulos em um unico arquivo .py")
    ap.add_argument("--src", default=here, help="pasta dos modulos (padrao: pasta deste script)")
    ap.add_argument("--modules", nargs="+", default=["pa_*.py"], help='padroes, ex.: "pa_*.py" "pb_*.py" "pc_*.py"')
    ap.add_argument("--entry", default="pa_gui_app.py", help="modulo de entrada (fica por ultimo)")
    ap.add_argument("--out", default="pa_unificado.py")
    ap.add_argument("--header", help="arquivo de texto com o cabecalho (nomes, matriculas, curso, campus)")
    a = ap.parse_args()

    paths = []
    for pat in a.modules:
        hits = sorted(glob.glob(os.path.join(a.src, pat)))
        if not hits:
            print(f"[aviso] nenhum arquivo para '{pat}' em {a.src}")
        paths += [h for h in hits if h not in paths]
    if not paths:
        sys.exit("nenhum modulo encontrado")

    if a.header:
        with open(a.header, encoding="utf-8") as f:
            header = "".join("# " + l if not l.startswith("#") else l for l in f.read().splitlines(True))
    else:
        header = HEADER_TEMPLATE
    try:
        src, ordered = build(paths, a.entry, header)
    except BundleError as e:
        sys.exit(f"ERRO: {e}")

    out = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print(f"Gerado: {out}  ({len(src.splitlines())} linhas, {len(src.encode('utf-8')) / 1024:.0f} KB)")
    print("Ordem:", " -> ".join(m.name for m in ordered))
    for m in ordered:
        if m.side_effects:
            print(f"[aviso] {m.name}.py executa codigo na importacao (linhas {m.side_effects[:6]}): "
                  f"confira se isso e esperado (ex.: registro de arquiteturas) ou embrulhe em funcao.")
    if "NOME COMPLETO" in src:
        print("[aviso] o cabecalho ainda tem o modelo: preencha nomes/matriculas/curso/campus e o numero do grupo "
              "(use --header cabecalho.txt).")
    errs = check(src, out)
    if errs:
        print("PROBLEMAS NO ARQUIVO GERADO:")
        for e in errs:
            print("  -", e)
        sys.exit(2)
    print("Checagens do arquivo gerado: OK (compila, sem imports internos, sem nomes indefinidos)")


if __name__ == "__main__":
    main()