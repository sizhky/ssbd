cleanup:
	find . -type f -name '*.ipynb' | grep -v '/.ipynb_checkpoints/' | tr '\n' ' ' | xargs nbstripout --drop-empty-cells
	nbdev_export
	black .
	nbdev_docs