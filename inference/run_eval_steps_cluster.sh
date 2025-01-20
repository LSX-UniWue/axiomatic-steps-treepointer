#! /bin/bash


mode=$3

if [ "$2" = "moresteps" ]
then
  evaluation_folder=evaluation_more_steps
  data_folder=trig_multistep_no_reverse_merged_more_steps/$mode
  beam_size=1
  end=9
fi
if [ "$2" = "standard" ]
then
  evaluation_folder=evaluation
  data_folder=trig_multistep_no_reverse_merged/small_test/$mode
  #beam_size=3
  beam_size=1
  end=6
fi
if [ "$2" = "deep" ]
then
  evaluation_folder=evaluation_deep
  data_folder=trig_multistep_no_reverse_merged_deep_test/$mode
  beam_size=1
  end=6
fi

checkpoints_folder=$1

mkdir -p /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_1
cp /eqseq/data/$data_folder/test.src /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_1
cp /eqseq/data/$data_folder/test.tgt /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_1
cp /eqseq/data/$data_folder/test.json /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_1/test.json

for ((i=1; i <= end; i++))
do
  if [ -f "/checkpoints/$checkpoints_folder/$evaluation_folder/results$i.csv" ]
  then
    echo "/checkpoints/$checkpoints_folder/$evaluation_folder/results$i.csv already exists"
    continue
  fi

  if [ "$mode" != "sequence" ]
  then
    python /checkpoints/preprocess_nstack2seq_merge.py --source-lang src --target-lang tgt --user-dir . --testpref /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/test --destdir /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/bin --joined-dictionary --no_remove_root --workers 8 --output-format binary --srcdict /eqseq/data/trig_multistep_no_reverse_merged/tree/bin/dict.src.txt --no_collapse
    python /checkpoints/generate.py /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/bin/ --path /checkpoints/$checkpoints_folder/checkpoint_best.pt --gen-subset test --task nstack_merge2seq --user-dir /checkpoints/src --append-eos-to-target --batch-size 1 --beam $beam_size --nbest $beam_size > /checkpoints/$checkpoints_folder/$evaluation_folder/results_step$i.txt
  else
    sed -i "s/SYMBOL\|CONSTANT\|[()]//g" /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/test.src
    fairseq-preprocess --source-lang src --target-lang tgt --user-dir . --testpref /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/test --destdir /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/bin --joined-dictionary --workers 8 --srcdict /eqseq/data/trig_multistep_no_reverse_merged/sequence/bin/dict.src.txt
    fairseq-generate /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/bin/ --path /checkpoints/$checkpoints_folder/checkpoint_best.pt --gen-subset test --task translation --user-dir /checkpoints/src --batch-size 1 --beam $beam_size --nbest $beam_size > /checkpoints/$checkpoints_folder/$evaluation_folder/results_step$i.txt
  fi

  python infer_multistep.py /checkpoints/$checkpoints_folder/$evaluation_folder/results_step$i.txt /checkpoints/$checkpoints_folder/$evaluation_folder/data_step_$i/test.json /checkpoints/$checkpoints_folder/$evaluation_folder/results$i.csv --axioms /eqseq/data/trig_multistep_no_reverse_merged/axioms.json
  python write_intermediate_trees.py /checkpoints/$checkpoints_folder/$evaluation_folder/results$i.csv $(($i+1))

done

python final_results.py /checkpoints/$checkpoints_folder/$evaluation_folder --max_beam 0