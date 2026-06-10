"""
Generate 50+ diverse documents across multiple domains for RAG evaluation.

Creates realistic knowledge base documents spanning different industries,
writing styles, and complexity levels to stress-test the RAG pipeline.
"""

import json
import os
import time
import argparse
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# 55 document specs across 11 domains (5 per domain)
DOCUMENT_SPECS = [
    # ── LEGAL ──
    {"filename": "terms_of_service.md", "domain": "legal",
     "title": "SaaSCorp Terms of Service",
     "prompt": "Write a Terms of Service for a B2B SaaS analytics platform called SaaSCorp. Include sections on: account eligibility (18+, business use), acceptable use policy, intellectual property rights, data ownership, limitation of liability ($500 cap), governing law (Delaware), dispute resolution (binding arbitration), termination conditions (30-day notice), and service level agreement (99.9% uptime, credits for downtime). Use specific numbers, dates, and thresholds throughout."},

    {"filename": "privacy_policy.md", "domain": "legal",
     "title": "SaaSCorp Privacy Policy",
     "prompt": "Write a Privacy Policy for SaaSCorp. Include: data collected (account info, usage analytics, cookies, device info), how data is used (service improvement, personalization, billing), third-party sharing (AWS hosting, Stripe billing, Mixpanel analytics), data retention (active accounts: indefinite, deleted accounts: 90-day purge), user rights (access, correction, deletion, export in CSV/JSON), GDPR compliance (lawful basis, DPO contact), CCPA compliance (opt-out of sale, categories disclosed), cookie policy (essential, analytics, marketing with specific cookie names and durations), and security measures (AES-256, TLS 1.3, SOC 2 Type II)."},

    {"filename": "data_processing_agreement.md", "domain": "legal",
     "title": "Data Processing Agreement",
     "prompt": "Write a Data Processing Agreement (DPA) for SaaSCorp as the data processor. Include: definitions (controller, processor, personal data, sub-processor), processing scope and purpose, processor obligations (only on documented instructions, confidentiality, security measures), sub-processors (list: AWS us-east-1, Stripe, SendGrid with notification requirements), data subject rights handling (72-hour response), breach notification (48 hours to controller), audit rights (annual, 30-day notice), data return/deletion (30 days after termination), international transfers (Standard Contractual Clauses), liability and indemnification."},

    {"filename": "acceptable_use_policy.md", "domain": "legal",
     "title": "SaaSCorp Acceptable Use Policy",
     "prompt": "Write an Acceptable Use Policy for SaaSCorp. Include: prohibited activities (illegal content, malware distribution, cryptocurrency mining, scraping, reverse engineering, competitive analysis), resource limits (API: 10K requests/hour, storage: 50GB per workspace, bandwidth: 100GB/month), account sharing rules (no credential sharing, SSO required for teams 5+), content standards (no hate speech, no adult content, no personally identifiable information of minors), enforcement (warning, suspension, termination timeline), reporting abuse (abuse@saascorp.com), and consequences of violation (immediate suspension for severe, 14-day cure period for minor)."},

    {"filename": "intellectual_property_policy.md", "domain": "legal",
     "title": "SaaSCorp Intellectual Property Policy",
     "prompt": "Write an Intellectual Property Policy. Include: customer data ownership (customer retains all rights), platform IP (SaaSCorp owns all platform code, algorithms, models), generated outputs (customer owns outputs generated using their data), feedback and suggestions (SaaSCorp may use without compensation), open source components (list 5 specific libraries with licenses: Apache 2.0, MIT), trademark usage (logo guidelines, co-marketing approval), API output restrictions (no training competing AI models), and content licensing (Creative Commons for documentation, proprietary for software)."},

    # ── HEALTHCARE ──
    {"filename": "patient_rights.md", "domain": "healthcare",
     "title": "MedFirst Clinic — Patient Rights & Responsibilities",
     "prompt": "Write a Patient Rights and Responsibilities document for MedFirst Clinic. Include: right to informed consent, right to refuse treatment, right to privacy (HIPAA protections), right to access medical records (30-day fulfillment, $0.10/page for paper, free electronic), right to second opinion, right to interpreter services (available in 12 languages), complaint process (Patient Advocate office, 5-business-day response), patient responsibilities (accurate medical history, timely payments, 24-hour cancellation notice), advance directives, non-discrimination policy, and visitor policy (2 visitors during business hours, 1 overnight)."},

    {"filename": "appointment_scheduling.md", "domain": "healthcare",
     "title": "MedFirst Clinic — Appointment & Scheduling Policy",
     "prompt": "Write an appointment scheduling policy. Include: scheduling methods (online portal, phone 8AM-6PM, walk-in for urgent), appointment types (new patient: 60 min, follow-up: 30 min, annual physical: 45 min, telehealth: 20 min), cancellation policy (24-hour notice required, $50 no-show fee after 2 occurrences), late arrival (15+ minutes may be rescheduled), wait time expectations (average 12 minutes, notification if 20+ minute delay), insurance verification (must provide 48 hours before new patient visits), referral requirements (specialist visits require PCP referral within 90 days), prescription refill process (48-hour advance request via portal, no weekend refills for controlled substances), and after-hours care (nurse hotline 24/7, ER for emergencies)."},

    {"filename": "billing_insurance_faq.md", "domain": "healthcare",
     "title": "MedFirst Clinic — Billing & Insurance FAQ",
     "prompt": "Write a billing and insurance FAQ. Include: accepted insurance plans (list 8 specific plan names), copay collection (due at time of visit), balance billing policy, payment plans (available for bills over $200, 0% interest for 12 months), financial assistance program (income below 200% FPL, application process), claim submission timeline (within 5 business days), explanation of benefits (how to read an EOB), appeals process (60 days to appeal denied claims, 3-level internal review), out-of-network benefits (60/40 coinsurance, $5000 deductible), and self-pay rates (20% discount for payment at time of service)."},

    {"filename": "telehealth_guide.md", "domain": "healthcare",
     "title": "MedFirst Clinic — Telehealth Guide",
     "prompt": "Write a telehealth guide. Include: eligible visit types (follow-ups, medication management, mental health, dermatology review, NOT physicals or procedures), technology requirements (smartphone/tablet/computer, camera, microphone, 5Mbps internet), platform (MedFirst Connect app, download from App Store/Google Play), setup instructions (account creation, identity verification, insurance card upload), pre-visit checklist (medication list, vitals if possible, well-lit quiet room), during visit (arrive 5 min early, have pharmacy info ready), prescriptions via telehealth (allowed for most non-controlled, e-prescribed to chosen pharmacy), follow-up (visit summary in portal within 24 hours, lab orders mailed or available at partner labs), technical troubleshooting (camera permissions, bandwidth test, fallback to phone call), and cost (same copay as in-person)."},

    {"filename": "hipaa_notice.md", "domain": "healthcare",
     "title": "MedFirst Clinic — HIPAA Notice of Privacy Practices",
     "prompt": "Write a HIPAA Notice of Privacy Practices. Include: uses and disclosures (treatment, payment, healthcare operations), patient rights (access records, request amendments, accounting of disclosures, request restrictions, confidential communications), required disclosures (law enforcement, public health, abuse reporting, judicial proceedings), minimum necessary standard, personal representatives, breach notification (60 days, written notice for 500+ individuals), complaint process (HHS Office for Civil Rights), effective date, contact information for Privacy Officer, and examples of permitted vs. prohibited disclosures."},

    # ── FINANCE/BANKING ──
    {"filename": "account_types.md", "domain": "banking",
     "title": "PinnacleBank — Account Types & Features",
     "prompt": "Write a banking account types document. Include 5 account types: Basic Checking (no minimum, $5/month fee waived with direct deposit), Premium Checking ($1500 minimum, free checks, ATM fee rebates up to $20/month), High-Yield Savings (3.75% APY, 6 transactions/month limit), Money Market ($10,000 minimum, tiered rates: 3.5%/3.75%/4.0% at 10K/50K/100K thresholds), Certificate of Deposit (terms: 6/12/18/24 months, rates: 4.0%/4.25%/4.5%/4.75%, early withdrawal penalty: 90/180 days interest). Include: FDIC insurance ($250,000 per depositor), overdraft protection options, and account comparison table details."},

    {"filename": "loan_products.md", "domain": "banking",
     "title": "PinnacleBank — Loan Products Guide",
     "prompt": "Write a loan products guide. Include: Personal Loans ($2,000-$50,000, 6.99%-15.99% APR, 24-60 month terms, no origination fee), Auto Loans (new: 5.49%-8.99%, used: 6.49%-10.99%, up to 84 months, 80% LTV for used), Home Equity Lines of Credit (variable rate prime+0.5%, $25,000-$500,000, 10-year draw period, 20-year repayment), Student Loan Refinancing (2.99%-7.49%, fixed and variable, 5-20 year terms), Small Business Loans (SBA 7(a) up to $5M, microloans up to $50K). For each: eligibility requirements, application process, approval timeline, required documents, and prepayment terms."},

    {"filename": "fraud_prevention.md", "domain": "banking",
     "title": "PinnacleBank — Fraud Prevention & Security Guide",
     "prompt": "Write a fraud prevention guide. Include: common fraud types (phishing, card skimming, check fraud, wire fraud, identity theft, account takeover), how to recognize them (specific red flag examples), what PinnacleBank does (real-time transaction monitoring, 2FA, card controls, $0 liability guarantee for unauthorized transactions reported within 60 days), what customers should do (strong passwords, monitor statements, freeze credit), reporting fraud (24/7 hotline 1-800-555-BANK, in-app reporting, branch visit), investigation timeline (provisional credit within 10 business days, resolution within 45 days), Zelle/P2P safety (authorized payments not reversible, verify recipient), travel notifications (set via app or call, cover up to 90 days), and card replacement (expedited 2-day delivery, temporary virtual card issued immediately)."},

    {"filename": "wire_transfer_guide.md", "domain": "banking",
     "title": "PinnacleBank — Wire Transfer Guide",
     "prompt": "Write a wire transfer guide. Include: domestic wires (cutoff 4PM ET for same-day, $25 outgoing fee, $15 incoming, requires ABA routing number and account number), international wires (cutoff 2PM ET, $45 outgoing, $15 incoming, requires SWIFT/BIC code, IBAN where applicable, intermediary bank info), processing times (domestic: same day if before cutoff, international: 1-3 business days), limits ($10,000 per day online, $50,000 in-branch with ID, $250,000 requires 24-hour pre-arrangement), required information checklist, cancellation policy (domestic: possible within 30 minutes, international: best effort, $25 recall fee), reporting requirements (CTR for $10,000+, SAR filing), and common errors to avoid."},

    {"filename": "online_banking_setup.md", "domain": "banking",
     "title": "PinnacleBank — Online & Mobile Banking Setup",
     "prompt": "Write an online banking setup guide. Include: enrollment (existing customers: SSN + account number + email, new customers: branch visit or video verification), mobile app (iOS 15+, Android 10+, download from official stores only), features (balance check, transfers, bill pay, mobile deposit up to $5,000/day, Zelle, card controls, branch/ATM locator), security setup (biometric login, 2FA via SMS or authenticator app, device registration, session timeout after 10 minutes), bill pay (one-time and recurring, 3 business days for electronic, 7 for check, payee limit of 50), mobile deposit (endorsement: 'For Mobile Deposit Only PinnacleBank', image quality requirements, holds: $225 available immediately, remainder next business day), alerts (low balance, large transaction, login from new device), and troubleshooting (locked account: wait 30 min or call, forgotten password: reset via email + security questions)."},

    # ── EDUCATION ──
    {"filename": "course_catalog.md", "domain": "education",
     "title": "Westfield University — Computer Science Course Catalog 2024-25",
     "prompt": "Write a CS course catalog. Include 15 courses across 4 levels: introductory (CS101 Intro to Programming, CS102 Data Structures, CS103 Discrete Math), intermediate (CS201 Algorithms, CS202 Operating Systems, CS203 Database Systems, CS204 Computer Networks), advanced (CS301 Machine Learning, CS302 Distributed Systems, CS303 Computer Security, CS304 Compiler Design), elective (CS401 Natural Language Processing, CS402 Computer Vision, CS403 Cloud Computing, CS404 Blockchain, CS405 Quantum Computing). For each: course code, title, credits (3 or 4), prerequisites, description (2-3 sentences), and which semester offered (Fall/Spring/Both). Include: credit requirements for BS (120 total, 45 CS, 15 math), and declaration requirements (complete CS201 with B or higher)."},

    {"filename": "grading_policy.md", "domain": "education",
     "title": "Westfield University — Academic Grading Policy",
     "prompt": "Write an academic grading policy. Include: grading scale (A: 93-100 4.0, A-: 90-92 3.7, B+: 87-89 3.3, B: 83-86 3.0, B-: 80-82 2.7, C+: 77-79 2.3, C: 73-76 2.0, C-: 70-72 1.7, D+: 67-69 1.3, D: 63-66 1.0, F: below 63 0.0), incomplete grades (request within last 2 weeks, 6-week completion deadline, becomes F if not completed), grade appeals (14 days from posting, written to department chair, committee review within 30 days), pass/fail option (max 2 per semester, not for major requirements, deadline: week 4), academic probation (GPA below 2.0, limited to 14 credits, required advisor meetings), Dean's List (semester GPA 3.5+, minimum 12 credits, no incompletes), Latin honors (summa: 3.9+, magna: 3.7+, cum laude: 3.5+), withdrawal policy (W grade through week 10, WF after), and repeat course policy (grade replacement once per course, max 3 total)."},

    {"filename": "financial_aid.md", "domain": "education",
     "title": "Westfield University — Financial Aid Handbook",
     "prompt": "Write a financial aid handbook. Include: application process (FAFSA by March 1, CSS Profile by February 15, verification documents within 30 days), types of aid (merit scholarships: $5K-$25K renewable, need-based grants: up to full tuition, federal Pell: up to $7,395, work-study: $3,000/year average, federal loans: $5,500-$7,500 depending on year), maintaining eligibility (2.0 GPA, 67% completion rate, 150% time frame), satisfactory academic progress (SAP appeal process, probation semester), outside scholarships (must be reported, may adjust need-based aid), study abroad funding (aid travels with student, consortium agreement required), summer aid (separate application, limited funding), cost of attendance breakdown (tuition: $48,500, room: $8,200, board: $6,800, books: $1,200, personal: $2,500, transportation: $1,500), and refund policy (100% before week 2, 75% week 2-3, 50% week 3-4, 0% after week 4)."},

    {"filename": "academic_integrity.md", "domain": "education",
     "title": "Westfield University — Academic Integrity Code",
     "prompt": "Write an academic integrity code. Include: definitions (plagiarism, cheating, fabrication, facilitation, unauthorized collaboration), AI policy (generative AI permitted only when explicitly allowed by instructor, must disclose use, AI-generated text without attribution is plagiarism), self-plagiarism (submitting same work in multiple courses without permission), citation standards (APA for social sciences, IEEE for CS/engineering, MLA for humanities), detection methods (Turnitin for papers, code similarity tools for programming, proctored exams via Respondus), sanctions (first offense: zero on assignment + academic integrity seminar, second offense: F in course, third offense: suspension or expulsion), reporting process (faculty report to Academic Integrity Board within 5 business days), student rights (written notification, hearing within 15 business days, right to advisor, appeal to Provost within 10 days), and record keeping (maintained for 7 years, expunged for first-offense graduates upon request)."},

    {"filename": "student_housing.md", "domain": "education",
     "title": "Westfield University — Student Housing Guide",
     "prompt": "Write a student housing guide. Include: residence halls (4 halls: Oak Hall freshmen-only 450 beds, Maple Hall sophomores 380 beds, Cedar Hall upperclass suite-style 200 beds, Pine Hall graduate 120 beds), room types and rates per semester (single: $5,200, double: $4,100, triple: $3,400, suite single: $5,800), meal plan requirement (freshmen: unlimited or 19/week, upperclass: optional, plans: unlimited $3,400, 14/week $2,800, 10/week $2,200, 50 block $900), housing application (deposit $300 non-refundable, lottery number by seniority, roommate matching questionnaire), move-in/move-out (fall: August 25-26, spring: January 12, checkout by 48 hours after last final), policies (quiet hours 10PM-8AM, no pets except fish under 10 gallons, no candles/incense, guests max 3 consecutive nights), maintenance requests (submit via housing portal, emergency: call 24/7 line, response: emergency 1 hour, urgent 24 hours, routine 5 business days), and early termination (medical/hardship only, pro-rated refund, $500 contract breakage fee)."},

    # ── IT/SECURITY ──
    {"filename": "security_policy.md", "domain": "it_security",
     "title": "CyberShield Inc — Information Security Policy",
     "prompt": "Write an information security policy. Include: classification levels (Public, Internal, Confidential, Restricted), handling requirements for each level, password requirements (16+ characters, no reuse of last 24, 90-day rotation, MFA mandatory for Confidential+), access control (principle of least privilege, quarterly access reviews, immediate revocation on termination), device security (full disk encryption, auto-lock 5 minutes, MDM enrollment required, personal devices: no Restricted data), network security (VPN required for remote access, WPA3 for WiFi, network segmentation between departments), incident classification (P1: data breach/system down, P2: degraded service, P3: minor issue, P4: informational), acceptable use (no personal cloud storage for work data, no unauthorized software, no public WiFi without VPN), and compliance requirements (SOC 2, ISO 27001, GDPR, HIPAA where applicable)."},

    {"filename": "incident_response_plan.md", "domain": "it_security",
     "title": "CyberShield Inc — Incident Response Plan",
     "prompt": "Write an incident response plan. Include 6 phases: 1. Preparation (team roles: Incident Commander, Technical Lead, Communications Lead, Legal Advisor; tools: SIEM, forensics toolkit, war room), 2. Identification (detection sources: SIEM alerts, employee reports, threat intel feeds; triage criteria; severity levels with response times: P1 15min, P2 1hr, P3 4hr, P4 next business day), 3. Containment (short-term: isolate affected systems, long-term: patch and harden, evidence preservation requirements), 4. Eradication (root cause analysis, malware removal, credential reset scope), 5. Recovery (system restoration from clean backups, monitoring period: 72 hours, sign-off requirements), 6. Lessons Learned (post-incident review within 5 business days, report template, metrics tracked). Include: communication templates, escalation matrix, and regulatory notification timelines (GDPR: 72 hours, HIPAA: 60 days, state breach laws: varies)."},

    {"filename": "vpn_setup_guide.md", "domain": "it_security",
     "title": "CyberShield Inc — VPN Setup & Troubleshooting Guide",
     "prompt": "Write a VPN setup guide. Include: supported clients (Windows: GlobalProtect 6.1+, macOS: GlobalProtect 6.1+, Linux: OpenConnect, iOS/Android: GlobalProtect app), installation steps for each platform (download URL, configuration: portal address vpn.cybershield.com, authentication: AD credentials + MFA push), split tunnel vs full tunnel (default: split tunnel for internal resources, full tunnel required for Restricted data access), connection profiles (Office: 192.168.0.0/16, Development: 10.0.0.0/8 + GitHub/AWS, Production: full tunnel + audit logging), troubleshooting (error codes: VPN-001 auth failed, VPN-002 certificate expired, VPN-003 split DNS failure, VPN-004 MTU mismatch, VPN-005 MFA timeout), performance (expected bandwidth: 80% of base connection, latency addition: 5-15ms domestic, 50-100ms international), and auto-connect policy (required when connecting to any non-corporate network)."},

    {"filename": "byod_policy.md", "domain": "it_security",
     "title": "CyberShield Inc — Bring Your Own Device Policy",
     "prompt": "Write a BYOD policy. Include: eligible devices (smartphones, tablets, laptops running Windows 10+, macOS 12+, iOS 16+, Android 12+), enrollment process (IT ticket, MDM profile installation, device compliance check), required security controls (screen lock: 6-digit PIN minimum or biometric, encryption: enabled, OS updates: within 14 days of release, antivirus: company-provided CrowdStrike), permitted activities (email, calendar, Slack, document viewing), prohibited activities (storing Confidential/Restricted data locally, jailbroken/rooted devices, USB debugging enabled), data separation (work profile container on Android, managed apps on iOS), remote wipe (company reserves right to wipe work container, full wipe only if device is lost/stolen and employee consents), privacy (company cannot access personal photos, messages, browsing history, or app usage), exit process (MDM removal within 24 hours of separation, work data wiped, personal data preserved), and support (IT supports work apps only, hardware issues are employee's responsibility)."},

    {"filename": "disaster_recovery.md", "domain": "it_security",
     "title": "CyberShield Inc — Disaster Recovery Plan",
     "prompt": "Write a disaster recovery plan. Include: RPO and RTO targets (Tier 1 critical: RPO 1hr/RTO 4hr, Tier 2 important: RPO 4hr/RTO 8hr, Tier 3 standard: RPO 24hr/RTO 48hr), backup strategy (daily incremental, weekly full, monthly archive, 3-2-1 rule: 3 copies, 2 media types, 1 offsite), system classification (Tier 1: production database, auth service, payment processing; Tier 2: email, CRM, internal tools; Tier 3: development environments, documentation), recovery procedures (failover to DR site in us-west-2, DNS update TTL 60 seconds, database replication lag monitoring), communication plan (internal: Slack #incident channel + PagerDuty, external: status page + customer email within 1 hour), testing schedule (tabletop exercises quarterly, partial failover semi-annually, full DR test annually), roles and responsibilities (DR Coordinator, Infrastructure Lead, Application Lead, Communications Lead), and post-recovery (data integrity verification, performance benchmarking, stakeholder sign-off)."},

    # ── REAL ESTATE ──
    {"filename": "lease_agreement_faq.md", "domain": "real_estate",
     "title": "Hartwell Properties — Residential Lease FAQ",
     "prompt": "Write a residential lease FAQ. Include: lease terms (standard 12-month, 6-month available at 10% premium, month-to-month after initial term at 15% premium), rent payment (due 1st of month, 5-day grace period, $50 late fee after day 5, $75 after day 10, 3 late payments in 12 months may trigger non-renewal), security deposit (1 month's rent, returned within 30 days of move-out, itemized deduction list provided, interest accrues at 1.5% annually in applicable states), utilities (tenant pays: electric, gas, internet; landlord pays: water, sewer, trash), pet policy (dogs and cats allowed, 2 pet maximum, $300 non-refundable pet fee + $35/month pet rent, breed restrictions: no aggressive breeds per insurance, weight limit 75 lbs), subletting (allowed with written landlord approval, $200 sublet processing fee, original tenant remains responsible), lease termination (60-day written notice, early termination fee: 2 months rent, military clause: SCRA compliant), and renewal process (notice sent 90 days before expiration, 30 days to respond, rent increase capped at 5% annually)."},

    {"filename": "maintenance_guide.md", "domain": "real_estate",
     "title": "Hartwell Properties — Maintenance Request Guide",
     "prompt": "Write a maintenance request guide. Include: how to submit (tenant portal at portal.hartwellprops.com, phone 8AM-5PM weekdays, emergency hotline 24/7), categories and response times (emergency: gas leak, fire, flood, no heat below 50°F — 2 hours; urgent: broken A/C above 90°F, no hot water, appliance failure — 24 hours; routine: leaky faucet, door adjustment, paint touch-up — 5 business days; cosmetic: minor scuffs, caulking — 10 business days), tenant responsibilities (replace air filters quarterly, smoke detector batteries, light bulbs, keeping drains clear, reporting issues promptly), landlord responsibilities (structural repairs, appliance replacement, HVAC maintenance, pest control for multi-unit), entry notice (24-hour written notice except emergencies, entry between 9AM-6PM weekdays), warranty on repairs (90 days on labor, manufacturer warranty on parts), seasonal maintenance schedule (spring: A/C inspection, fall: heating check, winter: pipe insulation), and what's NOT covered (damage caused by tenant negligence, unauthorized modifications, normal wear items under $25)."},

    {"filename": "move_in_checklist.md", "domain": "real_estate",
     "title": "Hartwell Properties — Move-In/Move-Out Guide",
     "prompt": "Write a move-in and move-out guide. Include: pre-move-in (lease signed 14+ days before, full deposit + first month due 7 days before, renter's insurance required: $100K liability minimum, utility transfers in tenant name by move-in date), move-in day (key pickup 9AM-5PM at management office, inspection checklist form, document existing damage with photos within 48 hours, mailbox key and parking permit issued), during tenancy (display unit number on door, no satellite dishes without approval, balcony restrictions: no grills, no storage, quiet hours 10PM-7AM), move-out (60-day written notice, schedule pre-inspection 2 weeks before, cleaning expectations: professional carpet cleaning required for pets, all personal items removed, holes patched, appliances cleaned), deposit deductions (common: carpet stains $150-300, wall damage $75-200 per wall, cleaning $200-400, key replacement $75, unreturned parking pass $50), and forwarding address (provide to office for deposit return and mail forwarding)."},

    {"filename": "community_rules.md", "domain": "real_estate",
     "title": "Hartwell Properties — Community Rules & Regulations",
     "prompt": "Write community rules for an apartment complex. Include: parking (1 assigned spot per unit, visitor parking in designated areas only, no commercial vehicles over 10,000 lbs, abandoned vehicles towed after 72 hours, $100 fine for parking violations), pool and amenities (pool hours 7AM-10PM, no glass containers, children under 12 must have adult supervision, gym hours 5AM-11PM, reserve party room 14 days in advance $150 deposit), noise (quiet hours 10PM-7AM weekdays, 11PM-8AM weekends, noise complaints: first verbal warning, second written warning, third $200 fine, fourth lease violation), trash and recycling (dumpster locations, recycling guidelines, bulk item pickup: schedule 48 hours ahead $25 fee, no hazardous materials), common areas (no personal items in hallways or stairwells, laundry room hours 7AM-10PM, report spills immediately), guest policy (temporary guests up to 14 consecutive days, beyond 14 days must be added to lease, background check required), and violations process (written notice, 10-day cure period, repeated violations: non-renewal at lease end)."},

    {"filename": "rental_application.md", "domain": "real_estate",
     "title": "Hartwell Properties — Rental Application Process",
     "prompt": "Write a rental application guide. Include: eligibility (gross income 3x monthly rent, credit score 620+, no evictions in past 5 years, no felony convictions in past 7 years), application fee ($50 non-refundable per applicant, all residents 18+ must apply), required documents (government-issued ID, 3 most recent pay stubs, 2 years tax returns for self-employed, bank statements if assets-based, landlord references for past 3 years), processing timeline (2-3 business days, up to 5 if verification delays), conditional approval (may require additional deposit of 1 month for credit scores 620-650, co-signer option available), guarantor requirements (credit score 700+, income 5x monthly rent, US-based), denial reasons (adverse credit history, insufficient income, negative landlord reference, incomplete application), fair housing compliance (no discrimination based on race, color, religion, sex, national origin, familial status, disability), and appeals process (written appeal within 10 days, provide supplemental documentation)."},

    # ── INSURANCE ──
    {"filename": "auto_insurance_coverage.md", "domain": "insurance",
     "title": "TrustGuard Insurance — Auto Insurance Coverage Guide",
     "prompt": "Write an auto insurance coverage guide. Include coverage types: liability (bodily injury: $50K/$100K state minimum, property damage: $25K minimum, recommended: $100K/$300K/$100K), collision (covers your vehicle in accidents, deductibles: $250/$500/$1000, diminishing deductible: reduces $50/year claim-free), comprehensive (theft, weather, animals, glass, deductibles: $100/$250/$500), uninsured/underinsured motorist ($50K/$100K recommended), medical payments ($5K-$25K), roadside assistance ($75/year, covers towing up to 25 miles, lockout, flat tire, fuel delivery), rental reimbursement ($40/day, 30-day max). Include: discount programs (safe driver 15%, multi-policy 10%, anti-theft 5%, good student 8%, defensive driving course 10%, paperless billing 3%), claims process (report within 24 hours, adjuster assigned within 1 business day, rental car within 2 hours if eligible), and factors affecting premium (age, driving record, vehicle type, location, credit score, annual mileage)."},

    {"filename": "homeowners_insurance.md", "domain": "insurance",
     "title": "TrustGuard Insurance — Homeowners Insurance Guide",
     "prompt": "Write a homeowners insurance guide. Include: coverage components (Dwelling: rebuilding cost, Other Structures: 10% of dwelling, Personal Property: 50-70% of dwelling, Loss of Use: 20% of dwelling, Personal Liability: $100K-$500K, Medical Payments: $1K-$5K), policy types (HO-3: most common open perils, HO-5: comprehensive, HO-6: condo, HO-8: older homes), what's NOT covered (flood, earthquake, normal wear, mold, sewer backup — all available as endorsements), deductible options ($1,000/$2,500/$5,000, hurricane deductible: 2-5% of dwelling), replacement cost vs actual cash value, personal property inventory (recommend video walkthrough, store off-site), scheduled personal property (jewelry over $1,500, art over $2,500, electronics over $5,000 — must be appraised), claims process (document damage immediately, prevent further damage, get estimates, adjuster visit within 3-5 business days), and annual review (update for renovations, home value changes, new purchases)."},

    {"filename": "claims_process.md", "domain": "insurance",
     "title": "TrustGuard Insurance — Claims Filing Guide",
     "prompt": "Write a claims filing guide. Include: when to file (any covered loss, even if unsure about coverage — let adjuster determine), how to file (mobile app: photo/video upload, phone: 1-800-555-TRUST 24/7, online portal, agent office), information needed (policy number, date/time of loss, description, photos/videos, police report number if applicable, other party info for auto claims), timeline (file within 30 days of loss, sooner is better), what happens after filing (claim number assigned immediately, adjuster contact within 24 hours, inspection within 3-5 business days, estimate within 10 business days, payment within 15 business days of agreement), disputes (request re-inspection, independent appraisal clause: each party hires appraiser, umpire decides if disagreement, cost split 50/50), payment methods (direct deposit 2-3 business days, check mailed 7-10 business days, two-party check if mortgage: requires lender endorsement), deductible (subtracted from claim payment, waived if other party is at fault in auto), and impact on premiums (first claim in 5 years: no surcharge via Claim Forgiveness, subsequent claims: up to 20% increase for 3 years)."},

    {"filename": "life_insurance_options.md", "domain": "insurance",
     "title": "TrustGuard Insurance — Life Insurance Options",
     "prompt": "Write a life insurance options guide. Include: term life (10/20/30 year terms, coverage $100K-$5M, level premiums, convertible to permanent within first 10 years, no cash value, lowest cost), whole life (permanent coverage, guaranteed cash value growth at 2-3%, dividends possible, level premiums, paid-up at 65 option, loan against cash value up to 90% at 5% interest), universal life (flexible premiums and death benefit, cash value earns current interest rate 4-5%, minimum premium required to maintain, can increase/decrease death benefit with evidence of insurability), indexed universal life (cash value tied to S&P 500 index, 0% floor/12% cap, participation rate 80-100%), application process (online quote, health questionnaire, medical exam for coverage over $500K: blood/urine/EKG, underwriting classes: Preferred Plus, Preferred, Standard, Substandard), beneficiary designation (primary and contingent, per stirpes vs per capita, irrevocable vs revocable), riders (accidental death: 2x benefit, waiver of premium: disability, child term rider: $10K per child, accelerated death benefit: 50% advance for terminal illness), and needs calculator formula (10-12x annual income, plus debts, plus education costs, minus existing coverage and assets)."},

    {"filename": "umbrella_policy.md", "domain": "insurance",
     "title": "TrustGuard Insurance — Personal Umbrella Policy Guide",
     "prompt": "Write an umbrella insurance guide. Include: what it covers (excess liability above auto/home limits, personal injury: libel, slander, false arrest, landlord liability, worldwide coverage), coverage amounts ($1M-$5M in $1M increments), underlying requirements (auto: $250K/$500K liability, home: $300K liability), cost ($200-$400/year for first $1M, $75-$100 per additional $1M), what's excluded (professional liability, intentional acts, business pursuits, contractual liability, workers compensation), who needs it (homeowners, landlords, pet owners, anyone with assets exceeding insurance limits, parents of teen drivers, boat/ATV owners, social media users, coaches/volunteers), real-world scenarios (dog bite lawsuit $850K, car accident injury $600K, defamation lawsuit $400K, pool accident $1.2M), and claims process (underlying policy pays first up to limit, umbrella covers the excess, single deductible per occurrence, defense costs included and do not reduce coverage limit)."},

    # ── TELECOM ──
    {"filename": "service_plans.md", "domain": "telecom",
     "title": "NovaTel — Wireless Service Plans",
     "prompt": "Write a wireless service plans document. Include 4 plans: Basic ($35/month: 5GB data, unlimited talk/text, 480p streaming, no hotspot), Standard ($55/month: 25GB data then throttled to 1Mbps, unlimited talk/text, 720p streaming, 5GB hotspot), Premium ($75/month: 50GB premium data, unlimited talk/text, 1080p streaming, 25GB hotspot, 5GB international data in 30 countries), Unlimited Elite ($90/month: truly unlimited premium data, unlimited talk/text, 4K streaming, 50GB hotspot, 10GB international in 65 countries, free subscription to StreamMax). Include: family discounts (line 2: -$10, line 3: -$15, line 4+: -$20 each), autopay discount ($5/month per line), device payment plans (0% APR over 24/36 months, requires credit check, early payoff allowed), number transfer process (keep your number, 2-4 hours, don't cancel old service first), 5G coverage (available in 350+ cities, mmWave in 50 cities, C-band nationwide), and taxes/fees (varies by location, estimated $5-8/month per line)."},

    {"filename": "billing_faq.md", "domain": "telecom",
     "title": "NovaTel — Billing FAQ",
     "prompt": "Write a telecom billing FAQ. Include: billing cycle (starts on activation date, 28-day cycle, bill generated 7 days before due date), payment methods (credit/debit, bank transfer, in-store cash, NovaTel Pay balance), late payment (15-day grace period, $10 late fee, service suspended after 30 days past due, sent to collections after 90 days, $25 reactivation fee), disputed charges (call within 60 days, provisional credit during investigation, resolution within 2 billing cycles), prorated charges (first and last bill prorated, mid-cycle plan changes prorated), international usage (rates vary by country, travel pass: $10/day unlimited in 200+ countries, data-only: $5/day 500MB), premium services (short codes, third-party charges: block via account settings or text STOP, refund for unauthorized charges within 90 days), bill credits (loyalty credit after 12 months: $5/month, referral credit: $50 per referral, outage credits: prorated for outages over 4 hours), and paperless billing ($3/month fee for paper bills, switch to paperless in app or account settings)."},

    {"filename": "device_troubleshooting.md", "domain": "telecom",
     "title": "NovaTel — Device Troubleshooting Guide",
     "prompt": "Write a device troubleshooting guide. Include: no signal/service (check airplane mode, restart device, remove/reinsert SIM, check outage map at status.novatel.com, reset network settings, contact if persists 24+ hours), slow data speeds (check data usage vs plan limit, speed test at fast.novatel.com, expected speeds: 5G 100-300Mbps, LTE 25-75Mbps, congestion during peak hours 5-9PM, disable VPN), can't make calls (check call forwarding settings, VoLTE enabled, try Wi-Fi calling if weak signal, check blocked numbers list, SIM card may need replacement after 3 years), text message issues (check message center number, clear message app cache, MMS requires data connection, group message settings, character limit: SMS 160, MMS 1600), voicemail setup (dial *86, create 4-7 digit PIN, greeting options: default/personal/name only, visual voicemail requires compatible device + data plan, transcription available on Standard+ plans), battery drain (check screen-on time, disable background app refresh, reduce screen brightness, check for rogue apps in battery settings, normal battery life: 8-12 hours active use), and when to visit a store vs call (store: hardware issues, SIM replacement, device trade-in; call/chat: billing, plan changes, account security, outage reports)."},

    {"filename": "trade_in_program.md", "domain": "telecom",
     "title": "NovaTel — Device Trade-In Program",
     "prompt": "Write a device trade-in guide. Include: eligibility (device must power on, no cracked screen, no water damage indicator tripped, battery holds charge, no activation lock/FRP lock), trade-in values (iPhone 14 Pro: up to $650, iPhone 13: up to $400, Samsung S23: up to $500, Samsung S22: up to $350, Google Pixel 8: up to $350, values decrease $25-50 quarterly), how to trade in (online: get instant quote, mail-in with prepaid label, in-store: immediate credit), promotional trade-ins (with new line activation: up to $1000 credit for qualifying devices, applied as bill credits over 24/36 months, cancel early = remaining balance due), condition tiers (Good: fully functional, minor cosmetic wear; Fair: functional with visible scratches/dents, 50% of Good value; Damaged: cracked screen or significant damage, 25% of Good value), pre-trade-in checklist (backup data, sign out of accounts, remove SIM and SD cards, disable Find My/Factory Reset Protection, factory reset), payment (in-store: instant account credit, mail-in: credit within 10 business days of receipt, promotional: bill credits start within 2-3 billing cycles), and price guarantee (if value drops within 14 days of quote, honored at original quote)."},

    {"filename": "international_roaming.md", "domain": "telecom",
     "title": "NovaTel — International Roaming Guide",
     "prompt": "Write an international roaming guide. Include: roaming options (Travel Pass: $10/day pay-per-use in 200+ countries, auto-charges only on days you use, International Plan: $50/month 15GB + unlimited talk/text in 65 countries, cruise ship: $15/day 1GB, in-flight WiFi: $8/flight on partner airlines), activation (Travel Pass: enabled by default on Premium+, add via app for other plans; International Plan: activate 24 hours before departure), country coverage tiers (Tier 1: Canada/Mexico included in all plans, Tier 2: Europe/Japan/Australia $10/day, Tier 3: other countries $15/day, check country-specific rates at novatel.com/international), data speeds abroad (LTE in most countries, throttled to 256Kbps after daily limit, no 5G roaming yet), calling rates without a plan (outgoing: $1.50-$3.50/min, incoming: $0.75-$1.50/min, texts: $0.50 sent, $0.05 received), tips (use Wi-Fi calling to avoid charges, download offline maps, disable background data for non-essential apps, set data usage alerts), emergency calls (911/112/999 always free worldwide), and returning home (roaming charges stop within 2 hours of connecting to domestic network, verify on app)."},

    # ── RESTAURANT/HOSPITALITY ──
    {"filename": "restaurant_menu_allergens.md", "domain": "hospitality",
     "title": "The Golden Plate — Menu & Allergen Guide",
     "prompt": "Write a restaurant menu with allergen information. Include 20 dishes across categories: Starters (5), Mains (8), Desserts (4), Beverages (3). For each dish: name, price, description (1-2 sentences), calories, and allergens present (from: gluten, dairy, eggs, nuts, soy, shellfish, fish, sesame). Include a cross-contamination disclaimer, how to request allergen-free preparation, and which dishes can be modified for dietary restrictions (vegetarian, vegan, gluten-free, keto). Example: 'Truffle Mushroom Risotto — $24 — Arborio rice slow-cooked with wild mushrooms, finished with truffle oil and aged parmesan. 680 cal. Contains: dairy, soy. Can be made vegan upon request (substitute nutritional yeast).'"},

    {"filename": "catering_services.md", "domain": "hospitality",
     "title": "The Golden Plate — Catering Services Guide",
     "prompt": "Write a catering services guide. Include: service tiers (Drop-Off: $25/person min 20 guests, Buffet: $45/person min 30 guests includes setup/cleanup, Full-Service: $85/person min 50 guests includes staff and equipment), menu packages (Corporate Lunch: 3 options $28-38/person, Wedding: customizable $75-150/person with tasting, Social Event: 3 options $35-55/person), booking process (inquiry form, consultation within 48 hours, tasting for events 50+ guests $150 credited to booking, contract and 25% deposit to confirm, final guest count 7 days before, final payment 3 days before), dietary accommodations (vegetarian, vegan, gluten-free, kosher, halal — notify 14 days in advance, 5% of guest count prepared as allergen-safe by default), rental equipment (tables: $12 each, chairs: $3 each, linens: $8 per table, china place setting: $5 per person, disposable eco-friendly: $2 per person), staffing (1 server per 20 guests buffet, 1 per 10 full-service, bartender: $250 per 4 hours), cancellation policy (30+ days: full refund minus $200 admin fee, 14-30 days: 50% refund, under 14 days: no refund, date changes: one free reschedule 30+ days out), and service area (within 30 miles of restaurant, $2/mile beyond, minimum $100 delivery fee)."},

    {"filename": "reservation_policy.md", "domain": "hospitality",
     "title": "The Golden Plate — Reservation & Dining Policy",
     "prompt": "Write a reservation and dining policy. Include: reservation methods (online at goldenplate.com, phone 11AM-9PM, OpenTable, walk-ins welcome based on availability), party size (2-6: standard reservation, 7-12: semi-private dining room $500 minimum spend, 13-20: private dining room $1500 minimum, 21+: full buyout or catering), cancellation (free cancellation up to 4 hours before, $25/person no-show fee charged to card on file, 3 no-shows: future reservations require prepaid deposit), late arrival (table held 15 minutes, after 15 minutes released, call to notify if running late), dress code (smart casual: collared shirts preferred, no athletic wear, no flip-flops, jacket required in Chef's Room), special occasions (birthday cake: $45 for 8-inch, $65 for 10-inch, order 48 hours ahead, complimentary dessert plate with 'Happy Birthday' plaque with reservation note), dietary needs (notify at booking, chef can customize any dish with 24-hour notice), children (kids menu available for under 12, high chairs available, no children under 5 in Chef's Room after 7PM), tipping (18% gratuity added for parties 6+, tip not included for smaller parties), and gift cards (available in $25-$500 denominations, no expiration, usable for dine-in and catering, purchase online or in-restaurant)."},

    {"filename": "loyalty_program.md", "domain": "hospitality",
     "title": "The Golden Plate — Loyalty Rewards Program",
     "prompt": "Write a loyalty program guide. Include: membership tiers (Bronze: free enrollment, Silver: after $500 annual spend, Gold: after $1500 annual spend, Platinum: after $3000 annual spend), earning points (1 point per $1 spent on food and beverage, 2x points on birthday month, 500 bonus points for reviews, 250 points for referrals), redemption (500 points = $5 credit, 2000 points = free appetizer, 4000 points = free entree up to $35, 10000 points = Chef's Table experience for 2), tier benefits (Bronze: birthday dessert + early access to seasonal menus; Silver: 10% off wine bottles + priority reservations; Gold: complimentary valet + quarterly tasting invitations + 15% off catering; Platinum: dedicated host + annual 4-course dinner for 2 + 20% off private events + first access to new locations), expiration (points expire after 12 months of inactivity, tier status resets annually on enrollment anniversary), how to join (app download, website, or ask your server, requires name and email), and fine print (points not earned on gift card purchases, alcohol excluded in some states, cannot combine with other offers except tier discounts, management reserves right to modify program with 30-day notice)."},

    {"filename": "food_safety.md", "domain": "hospitality",
     "title": "The Golden Plate — Food Safety & Sourcing Policy",
     "prompt": "Write a food safety and sourcing policy. Include: sourcing principles (local-first: 60% of produce from farms within 100 miles, seasonal menus updated quarterly, sustainable seafood: only MSC/ASC certified, meat: hormone-free and antibiotic-free, eggs: cage-free minimum, organic where available), food safety certifications (ServSafe certified kitchen, Grade A health department rating, annual third-party food safety audit), temperature control (cold storage: 40°F or below, freezer: 0°F, hot holding: 140°F minimum, cooking temps: poultry 165°F, ground meat 160°F, steaks 145°F, fish 145°F), allergen management (dedicated allergen-free prep station, color-coded cutting boards, staff allergen training quarterly, allergen binder available upon request), traceability (all proteins traceable to farm of origin, lot tracking for produce, recall response: affected items removed within 2 hours of notification), waste reduction (composting program, food donation partnership with local food bank, goal: 50% waste diversion by 2025, current: 38%), and customer incident reporting (allergic reaction: call 911, notify manager immediately, incident form completed within 1 hour, follow-up within 24 hours, insurance claim initiated within 48 hours)."},
]


def generate_document(spec: dict, model: str = "gemini/gemini-2.0-flash") -> str:
    """Generate a single document from a spec."""

    prompt = f"""{spec['prompt']}

IMPORTANT FORMATTING RULES:
- Use Markdown with clear headings (## for sections, ### for subsections)
- Include specific numbers, dates, thresholds, and dollar amounts throughout
- Write 800-1200 words
- Make it realistic — this should read like an actual company document
- Title: # {spec['title']}"""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate 50+ diverse documents")
    parser.add_argument("--output-dir", default="./documents",
                        help="Output directory for documents")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash",
                        help="LLM model for generation")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between API calls")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Resume from this document index")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(DOCUMENT_SPECS)
    print(f"Generating {total} documents across {len(set(s['domain'] for s in DOCUMENT_SPECS))} domains")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    generated = 0
    skipped = 0
    failed = 0

    for i, spec in enumerate(DOCUMENT_SPECS):
        if i < args.start_from:
            skipped += 1
            continue

        filepath = output_dir / spec["filename"]

        # Skip if already exists
        if filepath.exists():
            print(f"[{i+1}/{total}] SKIP (exists): {spec['filename']}")
            skipped += 1
            continue

        print(f"[{i+1}/{total}] Generating: {spec['filename']} [{spec['domain']}]...", end="", flush=True)

        content = generate_document(spec, args.model)

        if content:
            filepath.write_text(content, encoding="utf-8")
            word_count = len(content.split())
            print(f" ✓ ({word_count} words)")
            generated += 1
        else:
            print(" ✗ FAILED")
            failed += 1

        time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Done. Generated: {generated}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total documents in {output_dir}: {len(list(output_dir.glob('*.md')))}")

    # Print domain summary
    domains = {}
    for spec in DOCUMENT_SPECS:
        d = spec["domain"]
        domains[d] = domains.get(d, 0) + 1

    print(f"\nDomain breakdown:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count} documents")


if __name__ == "__main__":
    main()