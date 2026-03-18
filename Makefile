.PHONY: run-daily-pipeline

run-daily-pipeline:
	python quantpits/scripts/static_train.py --predict-only --all-enabled
	python quantpits/scripts/ensemble_fusion.py --from-config-all
	python quantpits/scripts/prod_post_trade.py
	python quantpits/scripts/order_gen.py
