#　differential learning  rate 在keras 上的实现与使用

differential learning rate 是指在训练的时候，不同的层使用不同的lr 去更新权重，这种训练的方法比较多的人使用，尤其在finetune的时候。这个在caffe上是可以直接在proto上写就可以了，但是在keras需要自己实现。

#### 1. 实现思路

- 不同的层有不同的lr，因此只需要在layer上增加一个mutilplier参数，让这个mutilplier乘以当前的lr，用于更新权重。然而并不是每个层都需要使用不同的lr, 　我们只需要在卷积层和全连接层使用不同的lr就可以了。
- 在layer上有了mutilplier这个参数后，我们在优化器上更新权重的时候，就把当前lr乘以这个mutilplier作为当前的lr，然后再更新权重。

#### 2. 更改的代码

- 我们希望`mutilplier`参数可以在定义层的时候作为参数传进去，这样使用起来比较方便。首先，我们需要在layer上增加`W_learning_rate_multiplier`和`b_learning_rate_multiplier`这两个参数，这两个参数是标量(scalar)，我们只在卷积层和全连接层增加这个参数就好了。

  在`convolutional.py`里，所有的卷积类都是`_Conv`这个类的子类，因此我们在这个基类上增加这个参数。

  ```python
  class _Conv(Layer):
      def __init__(self, rank,
                       filters,
                       kernel_size,
                       .
                   	 .
                   	 .
                       bias_constraint=None,
                       W_learning_rate_multiplier=None,#add 　
                   	 b_learning_rate_multiplier=None,#add
                       **kwargs):
          .
          .
          .
          self.W_learning_rate_multiplier = W_learning_rate_multiplier 
          self.b_learning_rate_multiplier = b_learning_rate_multiplier
  ```

  `_Conv`有了这两个参数后，我们把它作为一个字典存放，这个字典的映射关系是 参数=>multiplier，卷积核对应`W_learning_rate_multiplier`，偏置对应b_learning_rate_multiplier。这样在之后传参数就只需要传一个就好了。

  ```python 
  def build(self, input_shape):
      .
      .
      .
      self.multipliers = {}
      if self.W_learning_rate_multiplier is not None:
          self.multipliers[self.kernel] = self.W_learning_rate_multiplier
      if (self.bias is not None) and (self.b_learning_rate_multiplier is not None):
          self.multipliers[self.bias] = self.b_learning_rate_multiplier
  ```

  `keras`的每个层都有一个`config`用于保存这一层的所有参数，这个`config`是一个字典，我们需要把之前定义的`W_learning_rate_multiplier`和`b_learning_rate_multiplier`这两个参数也保存到这个字典里。

  ```python
  def get_config(self):
      config = {
        		.
         		.
              .
              'W_learning_rate_multiplier': self.W_learning_rate_multiplier if self.W_learning_rate_multiplier else None,
              'b_learning_rate_multiplier': self.b_learning_rate_multiplier if self.b_learning_rate_multiplier else None
          }
  ```

  这样的话，我们可以通过`_Conv`的对象得到这个`multipliers`。

  基类已经实现了，我们只需要在子类上的`__init__`增加这两个参数，并调用基类的`__init__`就可以把这些参数传进去并保存到变量里了。

  这里我们只贴出`Conv1D`这一个的更改，其他的比如`Conv2D`和`Conv3D`其实都是一样的。

  ```python
  ‘Conv1D’
  class Conv1D(_Conv):
      @interfaces.legacy_conv1d_support
      def __init__(self, filters,
                   kernel_size,
                   strides=1,
                   .
                   .
                   .
                   bias_constraint=None,
                   W_learning_rate_multiplier=None, 
                   b_learning_rate_multiplier=None,
                   **kwargs):
          super(Conv1D, self).__init__(
              rank=1,
              filters=filters,
              .
              .
              .
              bias_constraint=bias_constraint,
              W_learning_rate_multiplier = W_learning_rate_multiplier,
              b_learning_rate_multiplier = b_learning_rate_multiplier,
              **kwargs)       
  ```

  上面已经实现了在卷积层上增加`multiplier`这个参数，接下来我们在`Dense`层上也增加这个参数。做法其实还是一样的。

  ```python
  class Dense(Layer):
      def __init__(self, units,
                   activation=None,
                   .
                   .
                   .
                   bias_constraint=None,
                   W_learning_rate_multiplier=None,
                   b_learning_rate_multiplier=None,
                   **kwargs):
          .
          .
          .
          self.W_learning_rate_multiplier = W_learning_rate_multiplier
          self.b_learning_rate_multiplier = b_learning_rate_multiplier
      def build(self, input_shape):
          .
          .
          .
          self.multipliers = {}
          if self.W_learning_rate_multiplier is not None:
              self.multipliers[self.kernel] = self.W_learning_rate_multiplier
          if (self.bias is not None) and (self.b_learning_rate_multiplier is not None):
              self.multipliers[self.bias] = self.b_learning_rate_multiplier
      def get_config(self):
          config = {
              'units': self.units,
              .
              .
              .
              'W_learning_rate_multiplier': self.W_learning_rate_multiplier if self.W_learning_rate_multiplier else None,
              'b_learning_rate_multiplier': self.b_learning_rate_multiplier if self.b_learning_rate_multiplier else None
          }
  ```

- 接下来我们需要修改一些优化器的代码。`keras`所有的优化器都是继承与`optimizer`这个类，这个类有一个假的虚函数接口`get_updates`，这个写的比较假，也很有意思，可以看看：

  ```python
  @interfaces.legacy_get_updates_support
      def get_updates(self, loss, params):
          raise NotImplementedError
  ```

  显然所有继承它子类是必须实现这个函数的，这个函数就是更新权重的函数，我们希望更新的时候可以把`lr`乘以一个`multiplier`再去更新，因此我们在这个函数的的形参上增加这个字段。

  ```python
      def get_updates(self, loss, params, multipliers):
          raise NotImplementedError
  ```

  同样的，我们要在它的子类也要增加并实现将`lr`乘以这个`multipliers`用于更新权重。以下我们只贴`SGD`的实现。

  ```python
  class SGD(Optimizer):
      @interfaces.legacy_get_updates_support
      def get_updates(self, loss, params, multipliers):
          grads = self.get_gradients(loss, params)
          self.updates = [K.update_add(self.iterations, 1)]
          lr = self.lr
          if self.initial_decay > 0:
              lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                    K.dtype(self.decay))))
          # momentum
          shapes = [K.int_shape(p) for p in params]
          moments = [K.zeros(shape) for shape in shapes]
          self.weights = [self.iterations] + moments
          for p, g, m in zip(params, grads, moments):
              #如果参数在这个multiplier里，就取出这个来multiplier来
              if p in multipliers:
                  lrm = K.variable(multipliers[p])
              else:
                  lrm = K.variable(1.0)
              # print K.eval(lr*lrm)
              #乘以这个multiplier
              v = self.momentum * m - lr * lrm * g  # velocity
              self.updates.append(K.update(m, v))
              if self.nesterov:
                  #更新的时候乘以这个multiplier
                  new_p = p + self.momentum * v - lr * lrm * g
              else:
                  new_p = p + v

              # Apply constraints.
              if getattr(p, 'constraint', None) is not None:
                  new_p = p.constraint(new_p)

              self.updates.append(K.update(p, new_p))
          return self.updates
  ```

  ​

  优化器上我们已经实现了把带`multipliers`的层使用新的`lr`更新权重。在调用的优化器的时候我们传入这个参数。具体是在`training.py`里面的`Model`这个类，这个类的里`_make_train_function`里调用了更新权重的函数。首先我们要得到这个`multipliers`参数，我们需要在`Model`类里定义一个函数，用于得到这个参数:

  ```python
  class Model():
      .
      .
      .
      @property
      def multipliers(self):
          mults = {}
          for layer in self.layers:
              try:
                  for key, value in layer.multipliers.items():
                      if key in mults:
                          raise Exception('Received multiple learning rate multipliers '
                                          'for one weight tensor: ' + str(key))
                      mults[key] = value
              except:
                  pass
          return mults
  ```

  然后把这个参数传到优化器里的`get_updates`里：

  ```python
      def _make_train_function(self):
          if not hasattr(self, 'train_function'):
              raise RuntimeError('You must compile your model before using it.')
          self._check_trainable_weights_consistency()
          if self.train_function is None:
              inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
              if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                  inputs += [K.learning_phase()]

              with K.name_scope('training'):
                  with K.name_scope(self.optimizer.__class__.__name__):
   					# 将multipliers
                      training_updates = self.optimizer.get_updates(
                          params=self._collected_trainable_weights,
                          loss=self.total_loss, multipliers=self.multipliers)
                  updates = self.updates + training_updates
                  # Gets loss and metrics. Updates weights at each call.
                  self.train_function = K.function(inputs,
                                                   [self.total_loss] + self.metrics_tensors,
                                                   updates=updates,
                                                   name='train_function',
                                                   **self._function_kwargs)
  ```

  到此，就已经完成了。

- 上面已经实现在卷积层和全连接层使用`differential learning rate`。调用非常简单，给出两个例子：

  ```python
  #调用卷积
  x = Conv2D(filters=16, kernel_size=[2, 2], strides=[1, 1], padding='same',          W_learning_rate_multiplier=0.0001, b_learning_rate_multiplier=0.0001)(x)
  #调用全连接
  x = Dense(1024, activation='relu', W_learning_rate_multiplier=0.1, b_learning_rate_multiplier=0.1)(x)
  ```

  ​