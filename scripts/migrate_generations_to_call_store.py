"""Convert generation files (written by ff_modelgen) into records of the call
store of the model-call router, with origin "migrated". The source files are
not changed, and nothing is uploaded.

    python scripts/migrate_generations_to_call_store.py hf-dataset/data/v3 calls

See feedback_forensics.router.migrate.
"""

from feedback_forensics.router.migrate import main

if __name__ == "__main__":
    main()
