# CRM Data Dictionary & Notes

_Generated 2026-07-07_

This sample CRM dataset is internally consistent — the same `account_id`,
`deal_id`, `company`, and `owner` values link records across every file.

## Files
| File | Format | Grain | Key |
|------|--------|-------|-----|
| `deals.csv` | CSV | one row per opportunity | `deal_id` |
| `contacts.csv` | CSV | one row per person | `contact_id` |
| `accounts.json` | JSON | one object per company | `account_id` |
| `activities.json` | JSON | one row per touchpoint | `activity_id` |
| `crm_workbook.xlsx` | XLSX | 5 sheets + summary chart | — |
| `account_review.docx` | DOCX | narrative quarterly review | — |
| `deal_contracts.pdf` | PDF | closed-won contract summary | `deal_id` |

## Joins
- `deals.account_id` → `accounts[].account_id`
- `contacts.account_id` → `accounts[].account_id`
- `activities.deal_id` → `deals.deal_id`

## Snapshot metrics (for cross-checking answers)
- Total Closed Won: **$1,171,000**
- Open Pipeline: **$2,824,000**
- Win Rate: **65.4%**
- Deals: 60 total (17 won, 9 lost, 34 open)
- Accounts: 20 · Contacts: 40 · Leads: 25

## Good test questions for DataIntern
1. "What was total closed-won revenue?" → **$1,171,000** (deals.csv / xlsx)
2. "Who is the top rep by pipeline?" → check Summary sheet
3. "Does the PDF contract total match the deals sheet?" → cross-source
4. "List at-risk accounts" → account_review.docx
5. "Which lead sources convert best?" → leads.tsv + deals.csv
