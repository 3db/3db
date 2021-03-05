Creating a Custom Logger
========================

By default, 3DB has logging modules for bla, bla, and bla. We will now go
through the steps for implementing a custom logging module. To make things
clearer, we'll actually implement a new logger, ``CSVLogger``, that will write
out experimental results to a CSV file.

All it takes to implement a new logger is to subclass the provided base class,
:class:`threedb.result_logging.base_logger.BaseLogger`:

.. code-block:: python

    from threedb.result_logging.base_logger import BaseLogger

    class CSVLogger(BaseLogger):
        pass

In order to make this a valid logger, we need to provide implementations of two
abstract functions: ``__init__``, and ``log()``. The former 

.. code-block:: python

    def __init__(self, root_dir: str, result_buffer: BigChungusCyclicBuffer, config: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(root_dir, result_buffer, config)
        fname = path.join(root_dir, 'details.log')
        self.handle = open(fname, 'ab+')
        self.regid = self.buffer.register()
        self.evaluator = importlib.import_module(self.config['evaluation']['module']).Evaluator
        if 'label_map' in config['inference']:
        classmap_fname = path.join(root_dir, 'class_maps.json')

Next, we need to implement the ``log()`` function, which is called whenever a
new result is available to log.

.. code-block:: python

    def log(self, item):
        item = copy.deepcopy(item)
        rix = item['result_ix']
        # _, outputs, is_correct = self.result_buffer[rix]
        buffer_data = self.buffer[rix]
        item['output'] = buffer_data['output']
        item['is_correct'] = buffer_data['is_correct']
        item['output_type'] = self.evaluator.output_type
        cleaned = clean_log(item)
        encoded = json.dumps(cleaned, default=json_default, 
                             option=json.OPT_SERIALIZE_NUMPY | json.OPT_APPEND_NEWLINE)
        self.buffer.free(rix, self.regid)
        self.handle.write(encoded)

