.PHONY: run-daily-pipeline

run-daily-pipeline:
	python quantpits/scripts/prod_predict_only.py --all-enabled
	python quantpits/scripts/ensemble_fusion.py --from-config-all
	python quantpits/scripts/prod_post_trade.py
	python quantpits/scripts/order_gen.py
