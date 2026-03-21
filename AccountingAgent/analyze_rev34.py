"""Analyze rev34 logs to identify failing task patterns."""
import json
import re

with open(r'd:\NMiAI\AccountingAgent\rev34_logs.json', encoding='utf-8-sig') as f:
    data = json.load(f)

data.reverse()  # chronological order

# Find task boundaries
tasks = []
current = None
for e in data:
    msg = e.get('textPayload', '')
    ts = e.get('timestamp', '')[:19]
    if '=== New task received ===' in msg:
        if current:
            tasks.append(current)
        current = {'start': ts, 'msgs': [], 'prompt': '', 'files': '', 'errors': [], 'tool_calls': [], 'final': ''}
    if current:
        current['msgs'].append(msg)
        if 'Prompt' in msg and 'chars' in msg:
            current['prompt'] = msg
        if 'Files:' in msg:
            current['files'] = msg
        if 'Tool call:' in msg:
            current['tool_calls'].append(msg)
        if '_error' in msg or ('Error' in msg and 'status_code' in msg):
            current['errors'].append(msg)
        if 'Task completed' in msg:
            current['end_msg'] = msg
        if 'Agent final message' in msg:
            current['final'] = msg

if current:
    tasks.append(current)

print(f"Found {len(tasks)} tasks\n")

for i, t in enumerate(tasks):
    prompt = t.get('prompt', '')
    # Extract just the prompt text
    m = re.search(r'Prompt \(\d+ chars\): (.+)', prompt)
    prompt_text = m.group(1)[:150] if m else prompt[:150]
    
    end = t.get('end_msg', '')
    m2 = re.search(r'API calls: (\d+) \| Errors: (\d+)', end)
    calls = m2.group(1) if m2 else '?'
    errs = m2.group(2) if m2 else '?'
    
    m3 = re.search(r'completed in (\d+\.\d+)s', end)
    duration = m3.group(1) if m3 else '?'
    
    files = t.get('files', '')
    has_pdf = 'pdf' in files.lower() if files else False
    
    num_errors = len(t['errors'])
    
    # Categorize task
    pl = prompt_text.lower()
    if any(w in pl for w in ['ansatt', 'employee', 'empleado', 'empregado', 'employé', 'mitarbeiter', 'arbeidskontrak', 'tilbudsbrev', 'contrato']):
        cat = 'EMPLOYEE'
    elif any(w in pl for w in ['faktura', 'invoice', 'factura', 'fatura', 'facture', 'rechnung']):
        if any(w in pl for w in ['leverand', 'supplier', 'proveedor', 'fornecedor', 'fournisseur', 'lieferant']):
            cat = 'SUPPLIER_INV'
        elif any(w in pl for w in ['purre', 'reminder', 'mahnung', 'rappel']):
            cat = 'REMINDER_FEE'
        elif any(w in pl for w in ['kreditnota', 'credit note', 'nota de crédito', 'avoir', 'gutschrift']):
            cat = 'CREDIT_NOTE'
        elif any(w in pl for w in ['valuta', 'currency', 'kurs', 'taux']):
            cat = 'FX_INVOICE'
        else:
            cat = 'INVOICE'
    elif any(w in pl for w in ['reise', 'travel', 'viaje', 'viagem', 'voyage', 'reise']):
        cat = 'TRAVEL'
    elif any(w in pl for w in ['prosjekt', 'project', 'proyecto', 'projeto', 'projet', 'projekt']):
        if any(w in pl for w in ['livssyklus', 'lifecycle', 'ciclo', 'cycle']):
            cat = 'PROJECT_LIFECYCLE'
        else:
            cat = 'PROJECT'
    elif any(w in pl for w in ['avdeling', 'department', 'departamento', 'abteilung', 'département']):
        cat = 'DEPARTMENT'
    elif any(w in pl for w in ['dimensjon', 'dimension']):
        cat = 'DIMENSION'
    elif any(w in pl for w in ['lønn', 'salary', 'salario', 'salário', 'salaire', 'gehalt', 'lønnsslipp']):
        cat = 'SALARY'
    elif any(w in pl for w in ['bank', 'avstemming', 'reconcil']):
        cat = 'BANK_RECON'
    elif any(w in pl for w in ['bilag', 'voucher', 'kvittering', 'receipt']):
        cat = 'VOUCHER'
    elif any(w in pl for w in ['månedsslutt', 'periodeslutt', 'month-end', 'monatsende', 'periodiser']):
        cat = 'MONTH_END'
    elif any(w in pl for w in ['timar', 'timer', 'hours', 'horas', 'heures', 'stunden', 'timeregistrer']):
        cat = 'TIMESHEET'
    elif any(w in pl for w in ['analyse', 'analysis', 'análise', 'analys', 'kostn', 'livro razão', 'ledger', 'konti']):
        cat = 'LEDGER_ANALYSIS'
    elif any(w in pl for w in ['slett', 'delete', 'eliminar', 'supprimer', 'löschen', 'reverser', 'reverse']):
        cat = 'DELETE_REVERSE'
    elif any(w in pl for w in ['betaling', 'payment', 'pago', 'pagamento', 'paiement', 'zahlung']):
        cat = 'PAYMENT'
    elif any(w in pl for w in ['kunde', 'customer', 'cliente', 'client', 'kund']):
        cat = 'CUSTOMER'
    elif any(w in pl for w in ['produkt', 'product', 'producto', 'produit']):
        cat = 'PRODUCT'
    else:
        cat = 'OTHER'
    
    print(f"Task {i+1:2d} [{cat:18s}] {duration:>6s}s | calls={calls:>2s} errs={errs} | PDF={'Y' if has_pdf else 'N'}")
    print(f"         {prompt_text}")
    if num_errors > 0:
        for err in t['errors'][:3]:
            # Extract just the error detail
            m4 = re.search(r'"message":"([^"]+)"', err)
            if m4:
                print(f"         ERR: {m4.group(1)[:100]}")
            else:
                print(f"         ERR: {err[:120]}")
    print()
